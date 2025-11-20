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

// ---- CONFIG ----
const DATA_DIR = path.join(__dirname, "data/metadata"); // where batch_i.json live
const CLIPS_DIR = path.join(__dirname, "data"); // where videos live
// Singular pipeline log file
const LOG_FILE = path.join(DATA_DIR, "pipeline_logs.json");

// Ensure the pipeline log exists and load it
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

/**
 * Update the log based on the current detected filesystem state.
 * 
 * Logs are indexed by:
 *   log.batches[batchId].procedures[procedureType]
 *
 * procedureType ∈ { "grading", "retrain", "generate", "policy", "nextBatchJson" }
 */
async function updatePipelineLog({
  batchId,
  completed,
  isLargest,
  hasPolicy,
  hasNextBatchJson,
  retrainFlagPath,
  generateFlagPath,
  retrainDone,
  generateDone,
  canRetrain,
  canGenerate,
  canLoadNext,
  nextPolicyPath,
  nextBatchJsonPath,
}) {
  const log = await loadPipelineLog();
  const key = String(batchId);
  const now = new Date().toISOString();

  if (!log.batches[key]) {
    log.batches[key] = {
      lastUpdated: null,
      procedures: {},
    };
  }

  const batchEntry = log.batches[key];
  batchEntry.lastUpdated = now;
  batchEntry.procedures = batchEntry.procedures || {};

  // Grading / completion status
  batchEntry.procedures.grading = {
    type: "grading",
    completed,
    isLargest,
    lastChecked: now,
  };

  // Retrain procedure
  batchEntry.procedures.retrain = {
    type: "retrain",
    flagPath: retrainFlagPath,
    flagExists: retrainDone,
    canRetrain,
    lastChecked: now,
  };

  // Generate procedure
  batchEntry.procedures.generate = {
    type: "generate",
    flagPath: generateFlagPath,
    flagExists: generateDone,
    canGenerate,
    lastChecked: now,
  };

  // Policy checkpoint for next batch
  batchEntry.procedures.policy = {
    type: "policy",
    policyPath: nextPolicyPath,
    exists: hasPolicy,
    lastChecked: now,
  };

  // Next batch metadata JSON
  batchEntry.procedures.nextBatchJson = {
    type: "nextBatchJson",
    jsonPath: nextBatchJsonPath,
    exists: hasNextBatchJson,
    canLoadNext,
    lastChecked: now,
  };

  await savePipelineLog(log);
}

// Ensure there's an entry for this batch in the log
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
    // Backfill fields if missing
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

// Called by /api/pipeline-state to snapshot current filesystem-based status
async function updatePipelineStatus(batchId, status) {
  const now = new Date().toISOString();
  const { log, entry } = await getPipelineEntry(batchId);
  entry.status = { ...status, lastChecked: now };
  await savePipelineLog(log);
}

app.use(express.json());

// Serve static frontend
app.use(express.static(path.join(__dirname, "public")));

// Serve data and clip folders statically so the browser can load JSON/videos
app.use("/data/metadata", express.static(DATA_DIR));
app.use("/data", express.static(CLIPS_DIR));

// Utility: ensure batch_i_results.json exists and is up-to-date
async function ensureResults(batchId) {
  const batchFile = path.join(DATA_DIR, `batch_${batchId}.json`);
  const resultsFile = path.join(DATA_DIR, `batch_${batchId}_results.json`);

  // Read base batch file
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
    // If results file doesn't exist, create from batch with grades = "ungraded"
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

  // Make sure every pair has a grade, in case batch file was updated
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

  // Determine if all pairs are graded (no "ungraded")
  const completed =
    mergedPairs.length > 0 &&
    mergedPairs.every((p) => p.grade && p.grade !== "ungraded");

  // If completed changed or pairs were updated, rewrite the file
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
    // Ensure the in-memory object has the completed flag even if we didn't rewrite
    results.completed = completed;
  }

  return results;

}

// GET /api/batch/:batchId -> returns batch_i_results.json (created/updated if needed)
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

// GET /api/max-batch -> scan DATA_DIR for batch_*.json and return highest batch index
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

// Pipeline state for a given base batch
app.get("/api/pipeline-state/:batchId", async (req, res) => {
  const batchId = Number(req.params.batchId);
  if (Number.isNaN(batchId)) {
    return res.status(400).json({ error: "Invalid batchId" });
  }

  try {
    // 1) Find global max batch
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

    // 2) Load results to see if completed
    let completed = false;
    try {
      const results = await ensureResults(batchId);
      completed = !!results.completed;
    } catch (e) {
      completed = false;
    }

    // 3) Read "already triggered" state from central JSON
    const { entry } = await getPipelineEntry(batchId);
    const retrainDone = !!(entry.retrain && entry.retrain.triggered);
    const generateDone = !!(entry.generate && entry.generate.triggered);

    // 4) Check for next policy + next batch json (based on current batchId)
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

    // 5) Gating – logic is unchanged, just no .flag files anymore
    const canRetrain = isLargest && completed && !retrainDone;
    const canGenerate = isLargest && hasPolicy && !generateDone;
    const canLoadNext = hasNextBatchJson; // we allow this even if not largest anymore

    // 6) Update the single JSON log based on what we just detected
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


// Trigger train.sh once for a batch (global across all clients)
app.post("/api/retrain", async (req, res) => {
  const { batchId } = req.body || {};
  const base = Number(batchId);
  if (Number.isNaN(base)) {
    return res.status(400).json({ error: "Invalid batchId" });
  }

  const scriptPath = path.join(__dirname, "train.sh");

  try {
    // Don't allow more than once per batch, using central log
    const { entry } = await getPipelineEntry(base);
    if (entry.retrain && entry.retrain.triggered) {
      return res
        .status(400)
        .json({ error: "Retrain already triggered for this batch" });
    }

    // Fire-and-forget train.sh
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



// Trigger generate.sh once for a batch (global)
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

    // Fire-and-forget generate.sh
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


// POST /api/batch/:batchId/pair/:pairId/grade
// Body: { grade: "pair1_better" | "pair2_better" | "similar" | "not_comparable" | "ungraded" }
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
    // Ensure results file exists and is synced
    const results = await ensureResults(batchId);

    // Find pair by pair_id, fallback to index
    const numericPairId = Number(pairIdParam);
    let targetIdx = results.pairs.findIndex(
      (p, idx) =>
        (typeof p.pair_id === "number" && p.pair_id === numericPairId) ||
        (isNaN(numericPairId) && idx.toString() === pairIdParam)
    );
    if (targetIdx === -1 && !isNaN(numericPairId)) {
      // fallback: try index-based
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
    // Recompute completed: true if no pair is "ungraded"
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
