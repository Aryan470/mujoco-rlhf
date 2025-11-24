// server.js
const express = require("express");
const fs = require("fs/promises");
const path = require("path");
const { execFile } = require("child_process");

async function fileExists(p) {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

const app = express();
const PORT = 6006;

const DATA_DIR = path.join(__dirname, "data/metadata");
const CLIPS_DIR = path.join(__dirname, "data");
const LOG_FILE = path.join(DATA_DIR, "pipeline_logs.json");

async function loadPipelineLog() {
  try {
    const raw = await fs.readFile(LOG_FILE, "utf-8");
    return JSON.parse(raw);
  } catch (err) {
    if (err.code === "ENOENT") {
      const initial = { batches: {} };
      await fs.writeFile(LOG_FILE, JSON.stringify(initial, null, 2));
      return initial;
    }
    throw err;
  }
}

async function savePipelineLog(log) {
  await fs.writeFile(LOG_FILE, JSON.stringify(log, null, 2));
}


// async function updatePipelineLog({
//   batchId,
//   completed,
//   isLargest,
//   hasPolicy,
//   hasNextBatchJson,
//   retrainFlagPath,
//   generateFlagPath,
//   retrainDone,
//   generateDone,
//   canRetrain,
//   canGenerate,
//   canLoadNext,
//   nextPolicyPath,
//   nextBatchJsonPath,
// }) {
//   const log = await loadPipelineLog();
//   const key = String(batchId);
//   const now = new Date().toISOString();

//   if (!log.batches[key]) {
//     log.batches[key] = {
//       lastUpdated: null,
//       procedures: {},
//     };
//   }

//   const batchEntry = log.batches[key];
//   batchEntry.lastUpdated = now;
//   batchEntry.procedures = batchEntry.procedures || {};

//   batchEntry.procedures.grading = {
//     type: "grading",
//     completed,
//     isLargest,
//     lastChecked: now,
//   };

//   batchEntry.procedures.retrain = {
//     type: "retrain",
//     flagPath: retrainFlagPath,
//     flagExists: retrainDone,
//     canRetrain,
//     lastChecked: now,
//   };

//   batchEntry.procedures.generate = {
//     type: "generate",
//     flagPath: generateFlagPath,
//     flagExists: generateDone,
//     canGenerate,
//     lastChecked: now,
//   };

//   batchEntry.procedures.policy = {
//     type: "policy",
//     policyPath: nextPolicyPath,
//     exists: hasPolicy,
//     lastChecked: now,
//   };

//   batchEntry.procedures.nextBatchJson = {
//     type: "nextBatchJson",
//     jsonPath: nextBatchJsonPath,
//     exists: hasNextBatchJson,
//     canLoadNext,
//     lastChecked: now,
//   };

//   await savePipelineLog(log);
// }

async function getPipelineEntry(batchId) {
  const log = await loadPipelineLog();
  const key = String(batchId);

  if (!log.batches[key]) {
    log.batches[key] = {
      retrain: { triggered: false, lastTriggered: null, lastUpdated: null },
      generate: { triggered: false, lastTriggered: null, lastUpdated: null },
      status: null,
    };
  } else {
    if (!log.batches[key].retrain) {
      log.batches[key].retrain = {
        triggered: false,
        lastTriggered: null,
        lastUpdated: null,
      };
    }
    if (!log.batches[key].generate) {
      log.batches[key].generate = {
        triggered: false,
        lastTriggered: null,
        lastUpdated: null,
      };
    }
  }

  return { log, entry: log.batches[key], key };
}

async function markRetrainTriggered(batchId) {
  const now = new Date().toISOString();
  const { log, entry } = await getPipelineEntry(batchId);
  entry.retrain.triggered = true;
  entry.retrain.lastTriggered = now;
  entry.retrain.lastUpdated = now;
  await savePipelineLog(log);
}

async function markGenerateTriggered(batchId) {
  const now = new Date().toISOString();
  const { log, entry } = await getPipelineEntry(batchId);
  entry.generate.triggered = true;
  entry.generate.lastTriggered = now;
  entry.generate.lastUpdated = now;
  await savePipelineLog(log);
}

async function updatePipelineStatus(batchId, status) {
  const now = new Date().toISOString();
  const { log, entry } = await getPipelineEntry(batchId);
  entry.status = { ...status, lastChecked: now };
  await savePipelineLog(log);
}

app.use(express.json());

app.use(express.static(path.join(__dirname, "public")));

app.use("/data/metadata", express.static(DATA_DIR));
app.use("/data", express.static(CLIPS_DIR));

async function ensureResults(batchId) {
  const batchFile = path.join(DATA_DIR, `batch_${batchId}.json`);
  const resultsFile = path.join(DATA_DIR, `batch_${batchId}_results.json`);

  let batchRaw;
  try {
    batchRaw = await fs.readFile(batchFile, "utf-8");
  } catch (err) {
    if (err.code === "ENOENT") {
      throw new Error(`Batch file not found: batch_${batchId}.json`);
    }
    throw err;
  }
  const batch = JSON.parse(batchRaw);

  let results;
  try {
    const resultsRaw = await fs.readFile(resultsFile, "utf-8");
    results = JSON.parse(resultsRaw);
  } catch (err) {
    if (err.code === "ENOENT") {
      results = {
        batch_id: batch.batch_id,
        completed: false,
        pairs: batch.pairs.map((p, idx) => {
          const pairId = typeof p.pair_id === "number" ? p.pair_id : idx;
          const grader = ["Aryan", "Reina", "Beto"][pairId % 3];
          return {
            ...p,
            grade: "ungraded",
            grader
          };
        }),
      };
      await fs.writeFile(resultsFile, JSON.stringify(results, null, 2));
    } else {
      throw err;
    }
  }

  let updated = false;
  const pairMap = new Map();
  results.pairs.forEach((p, idx) => {
    const key =
      typeof p.pair_id === "number" ? p.pair_id : `idx_${idx}`;
    pairMap.set(key, p);
  });

  const mergedPairs = batch.pairs.map((p, idx) => {
    const key =
      typeof p.pair_id === "number" ? p.pair_id : `idx_${idx}`;
    const existing = pairMap.get(key);
    if (existing) {
      if (!("grade" in existing)) {
        updated = true;
        return { ...existing, grade: "ungraded" };
      }
      return existing;
    } else {
      updated = true;
      return { ...p, grade: "ungraded" };
    }
  });

  const completed =
    mergedPairs.length > 0 &&
    mergedPairs.every((p) => p.grade && p.grade !== "ungraded");

  if (!results || results.completed !== completed) {
    updated = true;
  }

  if (updated) {
    results = {
      batch_id: batch.batch_id,
      completed,
      pairs: mergedPairs,
    };
    await fs.writeFile(resultsFile, JSON.stringify(results, null, 2));
  } else {
    results.completed = completed;
  }

  return results;

}

app.get("/api/batch/:batchId", async (req, res) => {
  const batchId = req.params.batchId;
  try {
    const results = await ensureResults(batchId);
    res.json(results);
  } catch (err) {
    console.error(err);
    res.status(404).json({ error: err.message });
  }
});

app.get("/api/max-batch", async (req, res) => {
  try {
    const files = await fs.readdir(DATA_DIR);
    const batchRegex = /^batch_(\d+)\.json$/;
    let maxBatch = null;

    for (const file of files) {
      const match = batchRegex.exec(file);
      if (match) {
        const n = Number(match[1]);
        if (!Number.isNaN(n)) {
          maxBatch = maxBatch === null ? n : Math.max(maxBatch, n);
        }
      }
    }

    if (maxBatch === null) {
      return res.status(404).json({ error: "No batch files found" });
    }

    res.json({ maxBatch });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to scan batch files" });
  }
});

app.get("/api/pipeline-state/:batchId", async (req, res) => {
  const batchId = Number(req.params.batchId);
  if (Number.isNaN(batchId)) {
    return res.status(400).json({ error: "Invalid batchId" });
  }

  try {
    const files = await fs.readdir(DATA_DIR);
    const batchRegex = /^batch_(\d+)\.json$/;
    let maxBatch = null;
    for (const file of files) {
      const m = batchRegex.exec(file);
      if (m) {
        const n = Number(m[1]);
        if (!Number.isNaN(n)) {
          maxBatch = maxBatch === null ? n : Math.max(maxBatch, n);
        }
      }
    }

    const isLargest = maxBatch !== null && batchId === maxBatch;

    let completed = false;
    try {
      const results = await ensureResults(batchId);
      completed = !!results.completed;
    } catch (e) {
      completed = false;
    }

    const { entry } = await getPipelineEntry(batchId);
    const retrainDone = !!(entry.retrain && entry.retrain.triggered);
    const generateDone = !!(entry.generate && entry.generate.triggered);

    const nextPolicyPath = path.join(
      __dirname,
      "data",
      String(batchId + 1),
      "models",
      "checkpoints",
      "policy.pt"
    );
    const nextBatchJsonPath = path.join(DATA_DIR, `batch_${batchId + 1}.json`);

    const hasPolicy = await fileExists(nextPolicyPath);
    const hasNextBatchJson = await fileExists(nextBatchJsonPath);

    const canRetrain = isLargest && completed && !retrainDone;
    const canGenerate = isLargest && hasPolicy && !generateDone;
    const canLoadNext = hasNextBatchJson; // we allow this even if not largest anymore

    await updatePipelineStatus(batchId, {
      maxBatch,
      isLargest,
      completed,
      retrainDone,
      generateDone,
      hasPolicy,
      hasNextBatchJson,
      canRetrain,
      canGenerate,
      canLoadNext,
      nextBatchId: batchId + 1,
    });

    res.json({
      maxBatch,
      isLargest,
      completed,
      retrainDone,
      generateDone,
      hasPolicy,
      hasNextBatchJson,
      canRetrain,
      canGenerate,
      canLoadNext,
      nextBatchId: batchId + 1,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to compute pipeline state" });
  }
});


app.post("/api/retrain", async (req, res) => {
  const { batchId } = req.body || {};
  const base = Number(batchId);
  if (Number.isNaN(base)) {
    return res.status(400).json({ error: "Invalid batchId" });
  }

  const scriptPath = path.join(__dirname, "train.sh");

  try {
    const { entry } = await getPipelineEntry(base);
    if (entry.retrain && entry.retrain.triggered) {
      return res
        .status(400)
        .json({ error: "Retrain already triggered for this batch" });
    }

    execFile(scriptPath, { cwd: __dirname }, (err, stdout, stderr) => {
      if (err) {
        console.error("train.sh error:", err);
        console.error(stderr);
        return;
      }
      console.log("train.sh output:", stdout);
    });

    await markRetrainTriggered(base);

    res.json({ success: true });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to trigger retrain" });
  }
});



app.post("/api/generate-clips", async (req, res) => {
  const { batchId } = req.body || {};
  const base = Number(batchId);
  if (Number.isNaN(base)) {
    return res.status(400).json({ error: "Invalid batchId" });
  }

  const scriptPath = path.join(__dirname, "generate.sh");

  try {
    const { entry } = await getPipelineEntry(base);
    if (entry.generate && entry.generate.triggered) {
      return res
        .status(400)
        .json({ error: "Generate already triggered for this batch" });
    }

    execFile(scriptPath, { cwd: __dirname }, (err, stdout, stderr) => {
      if (err) {
        console.error("generate.sh error:", err);
        console.error(stderr);
        return;
      }
      console.log("generate.sh output:", stdout);
    });

    await markGenerateTriggered(base);

    res.json({ success: true });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to trigger generate" });
  }
});


app.post("/api/batch/:batchId/pair/:pairId/grade", async (req, res) => {
  const batchId = req.params.batchId;
  const pairIdParam = req.params.pairId;
  const { grade } = req.body;

  if (
    ![
      "pair1_better",
      "pair2_better",
      "similar",
      "not_comparable",
      "ungraded",
    ].includes(grade)
  ) {
    return res.status(400).json({ error: "Invalid grade value" });
  }

  const resultsFile = path.join(DATA_DIR, `batch_${batchId}_results.json`);

  try {
    const results = await ensureResults(batchId);

    const numericPairId = Number(pairIdParam);
    let targetIdx = results.pairs.findIndex(
      (p, idx) =>
        (typeof p.pair_id === "number" && p.pair_id === numericPairId) ||
        (isNaN(numericPairId) && idx.toString() === pairIdParam)
    );
    if (targetIdx === -1 && !isNaN(numericPairId)) {
      if (
        numericPairId >= 0 &&
        numericPairId < results.pairs.length
      ) {
        targetIdx = numericPairId;
      }
    }
    if (targetIdx === -1) {
      return res.status(404).json({ error: "Pair not found" });
    }

    results.pairs[targetIdx].grade = grade;
    const completed =
      results.pairs.length > 0 &&
      results.pairs.every((p) => p.grade && p.grade !== "ungraded");
    results.completed = completed;

    await fs.writeFile(resultsFile, JSON.stringify(results, null, 2));

    res.json({ success: true, completed });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to update grade" });
  }
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
