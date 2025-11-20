// server.js
const express = require("express");
const fs = require("fs/promises");
const path = require("path");

const app = express();
const PORT = 6006;

// ---- CONFIG ----
const DATA_DIR = path.join(__dirname, "data/metadata"); // where batch_i.json live
const CLIPS_DIR = path.join(__dirname, "data"); // where videos live

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
        pairs: batch.pairs.map((p) => ({
          ...p,
          grade: "ungraded",
        })),
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

  if (updated) {
    results = {
      batch_id: batch.batch_id,
      pairs: mergedPairs,
    };
    await fs.writeFile(resultsFile, JSON.stringify(results, null, 2));
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
    await fs.writeFile(resultsFile, JSON.stringify(results, null, 2));

    res.json({ success: true });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to update grade" });
  }
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
