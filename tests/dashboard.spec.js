const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { expect, test } = require('@playwright/test');

const dashboardUrl = pathToFileURL(
  path.join(__dirname, '..', 'dashboard', 'index.html')
).href;

test.describe('dashboard visual smoke', () => {
  test('renders without the sentiment donut covering other panels', async ({ page }) => {
    await page.goto(dashboardUrl);

    await expect(
      page.getByRole('heading', { name: 'Airline Passenger Satisfaction' })
    ).toBeVisible();
    await expect(page.locator('.donut')).toBeVisible();

    const donutMetrics = await page.locator('.donut').evaluate((element) => {
      const elementStyle = window.getComputedStyle(element);
      const beforeStyle = window.getComputedStyle(element, '::before');
      const rect = element.getBoundingClientRect();

      return {
        position: elementStyle.position,
        overflow: elementStyle.overflow,
        width: rect.width,
        beforeWidth: Number.parseFloat(beforeStyle.width)
      };
    });

    expect(donutMetrics.position).toBe('relative');
    expect(donutMetrics.overflow).toBe('hidden');
    expect(donutMetrics.width).toBeLessThanOrEqual(300);
    expect(donutMetrics.beforeWidth).toBeLessThanOrEqual(donutMetrics.width);

    const viewport = page.viewportSize();
    if (viewport && viewport.width >= 900) {
      const monthlyPanel = page.locator('.panel', {
        hasText: 'Monthly Satisfaction Pulse'
      });
      const panelBox = await monthlyPanel.boundingBox();
      expect(panelBox).not.toBeNull();

      const samplePoint = {
        x: panelBox.x + panelBox.width - 80,
        y: panelBox.y + panelBox.height / 2
      };

      const hitTest = await page.evaluate(({ x, y }) => {
        const element = document.elementFromPoint(x, y);
        return {
          inDonut: Boolean(element && element.closest('.donut')),
          className: String(element && element.className)
        };
      }, samplePoint);

      expect(hitTest.inDonut).toBe(false);
    }
  });
});
