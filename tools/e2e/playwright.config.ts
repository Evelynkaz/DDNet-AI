import { defineConfig } from "@playwright/test";

// A real-browser smoke test for task 5.1's login/status/logout flow (see README.md in this
// folder). Not part of CI: it needs a chromium-headless-shell install and drives a real
// `ddnet-ai web` process, which the workspace's normal `cargo test` runs never do.
export default defineConfig({
  testDir: ".",
  timeout: 30_000,
  // The page is drawn in software WebGL on a shared machine and logins use the production password hash: assertions wait longer.
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  reporter: "list",
  use: {
    headless: true,
    // Task 5.10: the «Игра» tab draws with WebGL2. A headless machine has no GPU, so Chromium rasterises in software (SwiftShader),
    // which newer versions only allow with this flag; the last flag lets the page (loopback) be a fulfilled/route-served document.
    launchOptions: {
      args: [
        "--use-gl=angle",
        "--use-angle=swiftshader",
        "--enable-unsafe-swiftshader",
        "--ignore-gpu-blocklist",
        "--disable-features=LocalNetworkAccessChecks,PrivateNetworkAccessPermissionPrompt,BlockInsecurePrivateNetworkRequests",
      ],
    },
  },
});
