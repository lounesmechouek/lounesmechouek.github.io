const markdownIt = require("markdown-it");
const markdownItAnchor = require("markdown-it-anchor");

module.exports = function (eleventyConfig) {
  // Markdown configuration
  const md = markdownIt({
    html: true,
    breaks: false,
    linkify: true,
    typographer: true,
  }).use(markdownItAnchor);

  eleventyConfig.setLibrary("md", md);

  // Pass through assets
  eleventyConfig.addPassthroughCopy("src/assets/images");
  eleventyConfig.addPassthroughCopy("src/writing/**/*.{png,jpg,jpeg,gif,svg,webp}");

  // Collection: all writing posts sorted by date
  eleventyConfig.addCollection("posts", function (collectionApi) {
    return collectionApi
      .getFilteredByGlob("src/writing/**/*.md")
      .filter((item) => !item.data.draft)
      .sort((a, b) => b.date - a.date);
  });

  // Filter: format date
  eleventyConfig.addFilter("dateFormat", function (date, format) {
    const d = new Date(date);
    const months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];
    if (format === "short") {
      return `${months[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
    }
    if (format === "month-year") {
      return `${months[d.getMonth()]} ${d.getFullYear()}`;
    }
    return d.toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
  });

  // Filter: reading time estimate
  eleventyConfig.addFilter("readingTime", function (content) {
    const words = (content || "").split(/\s+/).length;
    return Math.max(1, Math.ceil(words / 250));
  });

  // Filter: limit array
  eleventyConfig.addFilter("limit", function (arr, limit) {
    return (arr || []).slice(0, limit);
  });

  return {
    dir: {
      input: "src",
      output: "_site",
      includes: "_includes",
      data: "_data",
    },
    templateFormats: ["njk", "md", "html"],
    markdownTemplateEngine: "njk",
    htmlTemplateEngine: "njk",
  };
};
