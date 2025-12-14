const express = require("express");
const axios = require("axios");
const path = require("path");
const { marked } = require("marked");

const app = express();

/* -----------------------------
   Middleware
----------------------------- */
app.use(express.urlencoded({ extended: true }));
app.use(express.static("public"));

app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));

/* -----------------------------
   Utility: Normalize Markdown
   (Fixes inline bullets from LLM)
----------------------------- */
function normalizeMarkdown(text) {
  if (!text) return "";

  return text
    // ensure bullets start on new lines
    .replace(/\s*\*\s+\*\*/g, "\n* **")
    // clean excessive newlines
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/* -----------------------------
   Routes
----------------------------- */

// Home
app.get("/", (req, res) => {
  res.render("index");
});

// Search
app.post("/search", async (req, res) => {
    const query = req.body.query;
  
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/search-rag",
        { query }
      );
  
      const rawAnswer = response.data.answer;
      const normalized = normalizeMarkdown(rawAnswer);
      const answerHtml = marked.parse(normalized);
  
      // ✅ IMPORTANT: answerHtml is passed here
      res.render("results", {
        query: query,
        answerHtml: answerHtml,
        products: response.data.products
      });
  
    } catch (error) {
      console.error("Search error:", error.message);
      res.status(500).send("Something went wrong while fetching results.");
    }
  });
  

/* -----------------------------
   Server Start
----------------------------- */
app.listen(3000, () => {
  console.log("✅ UI running at http://localhost:3000");
});
