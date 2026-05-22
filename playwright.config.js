const { defineConfig, devices } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests',
  timeout: 30000,
  retries: 0,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    screenshot: 'only-on-failure',
    trace: 'on-first-retry'
  },
  projects: [
    {
      name: 'chromium-desktop',
      use: {
        browserName: 'chromium',
        viewport: { width: 1366, height: 900 }
      }
    },
    {
      name: 'chromium-mobile',
      use: {
        browserName: 'chromium',
        ...devices['Pixel 7']
      }
    }
  ]
});
