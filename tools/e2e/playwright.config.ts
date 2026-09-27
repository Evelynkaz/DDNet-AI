import { defineConfig } from "@playwright/test";

// A real-browser smoke test for task 5.1's login/status/logout flow (see README.md in this
// folder). Not part of CI: it needs a chromium-headless-shell install and drives a real
// `ddnet-ai web` process, which the workspace's normal `cargo test` runs never do.
export default defineConfig({
  testDir: ".",
  timeout: 30_000,
  fullyParallel: false,
  workers: 1,
  reporter: "list",
  use: {
    headless: true,
  },
});
