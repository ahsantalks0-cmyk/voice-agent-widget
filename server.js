import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
const HOST = "0.0.0.0";

// Enable CORS for embeddable widget script and assets
app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "*");
  next();
});

// Serve root files (such as /widget.js)
app.use(express.static(__dirname));

// Serve widget-host directory
app.use(express.static(path.join(__dirname, "widget-host")));

// Root fallback to widget-host/index.html
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "widget-host", "index.html"));
});

app.listen(PORT, HOST, () => {
  console.log(`Voice Agent Widget server running on http://${HOST}:${PORT}`);
});
