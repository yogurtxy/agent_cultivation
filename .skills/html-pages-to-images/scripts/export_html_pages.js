#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const { pathToFileURL } = require("url");

function parseArgs(argv) {
  const options = {
    selector: ".page",
    width: 1080,
    height: 1440,
    scale: 2,
    format: "png",
    quality: 92,
    wait: 250,
    showControls: false,
    clipSelector: "auto",
  };
  const positional = [];

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith("--")) {
      positional.push(arg);
      continue;
    }
    const key = arg.slice(2);
    if (key === "show-controls") {
      options.showControls = true;
      continue;
    }
    if (key === "no-clip") {
      options.clipSelector = "";
      continue;
    }
    const value = argv[i + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`Missing value for --${key}`);
    }
    i += 1;
    if (key === "output-dir") options.outputDir = value;
    else if (key === "selector") options.selector = value;
    else if (key === "width") options.width = Number(value);
    else if (key === "height") options.height = Number(value);
    else if (key === "scale") options.scale = Number(value);
    else if (key === "format") options.format = value.toLowerCase();
    else if (key === "quality") options.quality = Number(value);
    else if (key === "wait") options.wait = Number(value);
    else if (key === "clip-selector") options.clipSelector = value;
    else throw new Error(`Unknown option --${key}`);
  }

  if (positional.length !== 1) {
    throw new Error("Usage: node export_html_pages.js <input.html> [--output-dir dir] [--selector .page] [--clip-selector .page] [--no-clip]");
  }
  if (!["png", "jpeg"].includes(options.format)) {
    throw new Error("--format must be png or jpeg");
  }
  if (!Number.isFinite(options.width) || !Number.isFinite(options.height) || options.width < 320 || options.height < 320) {
    throw new Error("--width and --height must be numbers >= 320");
  }
  if (!Number.isFinite(options.scale) || options.scale < 1 || options.scale > 4) {
    throw new Error("--scale must be a number between 1 and 4");
  }
  options.input = positional[0];
  return options;
}

function loadPlaywright() {
  try {
    return require("playwright");
  } catch (error) {
    try {
      return require(require.resolve("playwright", { paths: [process.cwd()] }));
    } catch (cwdError) {
      console.error("Playwright is required to render HTML screenshots.");
      console.error("Install it in the current project with:");
      console.error("  npm install playwright");
      console.error("  npx playwright install chromium");
      process.exit(2);
    }
  }
}

function cleanOutputDir(outputDir) {
  fs.mkdirSync(outputDir, { recursive: true });
  for (const entry of fs.readdirSync(outputDir)) {
    if (/^page-\d+\.(png|jpe?g)$/i.test(entry)) {
      fs.unlinkSync(path.join(outputDir, entry));
    }
  }
}

async function waitForFonts(page) {
  await page.evaluate(async () => {
    if (document.fonts && document.fonts.ready) {
      await document.fonts.ready;
    }
  });
}

async function resolvePageSelector(page, requestedSelector) {
  const requestedCount = await page.locator(requestedSelector).count();
  if (requestedCount) {
    return { selector: requestedSelector, count: requestedCount };
  }
  if (requestedSelector === ".page") {
    const slideCount = await page.locator(".slide").count();
    if (slideCount) {
      return { selector: ".slide", count: slideCount };
    }
  }
  return { selector: requestedSelector, count: 0 };
}

async function preparePageCapture(page, selector) {
  await page.addStyleTag({
    content: `
      [data-export-page-state="hidden"] {
        display: none !important;
        visibility: hidden !important;
      }
      [data-export-page-state="active"] {
        display: block !important;
        visibility: visible !important;
      }
    `,
  });
  await page.evaluate((pageSelector) => {
    for (const item of document.querySelectorAll(pageSelector)) {
      item.classList.remove("active");
      item.removeAttribute("aria-hidden");
      item.removeAttribute("data-export-page-state");
    }
  }, selector);
}

async function activatePage(page, selector, pageIndex) {
  await page.evaluate(
    ({ pageSelector, index }) => {
      const pages = Array.from(document.querySelectorAll(pageSelector));
      pages.forEach((item, itemIndex) => {
        const isActive = itemIndex === index;
        item.classList.toggle("active", isActive);
        item.setAttribute("aria-hidden", isActive ? "false" : "true");
        item.setAttribute("data-export-page-state", isActive ? "active" : "hidden");
      });
      window.scrollTo(0, 0);
      if (window.history && window.history.replaceState) {
        window.history.replaceState(null, "", `#page-${index + 1}`);
      }
    },
    { pageSelector: selector, index: pageIndex }
  );
}

async function capture() {
  const options = parseArgs(process.argv.slice(2));
  const inputPath = path.resolve(options.input);
  if (!fs.existsSync(inputPath)) {
    throw new Error(`Input file not found: ${inputPath}`);
  }

  const outputDir = path.resolve(
    options.outputDir || path.join(path.dirname(inputPath), `${path.basename(inputPath, path.extname(inputPath))}_images`)
  );
  cleanOutputDir(outputDir);

  const { chromium } = loadPlaywright();
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: options.width, height: options.height },
    deviceScaleFactor: options.scale,
  });
  const page = await context.newPage();

  try {
    await page.goto(pathToFileURL(inputPath).href, { waitUntil: "networkidle" });
    await waitForFonts(page);

    if (!options.showControls) {
      await page.addStyleTag({
        content: `
          .controls,
          nav[aria-label*="控制"],
          nav[aria-label*="Slide"],
          nav[aria-label*="slide"] {
            display: none !important;
            visibility: hidden !important;
          }
        `,
      });
    }

    const resolved = await resolvePageSelector(page, options.selector);
    const count = resolved.count;
    const pageSelector = resolved.selector;
    const clipSelector = options.clipSelector === "auto" ? pageSelector : options.clipSelector;
    const total = count || 1;
    const pad = String(total).length < 3 ? 3 : String(total).length;
    const written = [];

    if (count) {
      await preparePageCapture(page, pageSelector);
    }

    for (let index = 0; index < total; index += 1) {
      if (count) {
        await activatePage(page, pageSelector, index);
      }

      await page.waitForTimeout(options.wait);
      const fileName = `page-${String(index + 1).padStart(pad, "0")}.${options.format === "jpeg" ? "jpg" : "png"}`;
      const outputPath = path.join(outputDir, fileName);
      const screenshotOptions = {
        path: outputPath,
        type: options.format,
        fullPage: false,
      };
      if (options.format === "jpeg") {
        screenshotOptions.quality = options.quality;
      }
      if (clipSelector) {
        await page.locator(clipSelector).nth(count ? index : 0).screenshot(screenshotOptions);
      } else {
        await page.screenshot(screenshotOptions);
      }
      written.push(outputPath);
    }

    console.log(`Exported ${written.length} image(s) to ${outputDir}`);
    for (const file of written) console.log(file);
  } finally {
    await browser.close();
  }
}

capture().catch((error) => {
  console.error(error.message || error);
  process.exit(1);
});
