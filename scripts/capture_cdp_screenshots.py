"""
scripts/capture_cdp_screenshots.py — Automated verification and screenshot capture
using Google Chrome DevTools Protocol (CDP) and websockets.

Captures the 5 required states:
1. Empty page
2. FAKE result
3. REAL result
4. UNCERTAIN result
5. Error state (empty input / non-English input)
"""
import os
import sys
import json
import time
import base64
import asyncio
import subprocess
import urllib.request
import websockets

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "reports", "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9222
USER_DATA_DIR = "/tmp/chrome_cdp_profile_truthpulse"

class CDPClient:
    def __init__(self, ws_url):
        self.ws_url = ws_url
        self.ws = None
        self.msg_id = 0

    async def connect(self):
        self.ws = await websockets.connect(self.ws_url, max_size=25 * 1024 * 1024)

    async def send_cmd(self, method, params=None):
        self.msg_id += 1
        payload = {"id": self.msg_id, "method": method, "params": params or {}}
        await self.ws.send(json.dumps(payload))
        while True:
            resp = await self.ws.recv()
            data = json.loads(resp)
            if data.get("id") == self.msg_id:
                if "error" in data:
                    raise RuntimeError(f"CDP error: {data['error']}")
                return data.get("result", {})

    async def evaluate_js(self, expression, await_promise=False):
        res = await self.send_cmd("Runtime.evaluate", {
            "expression": expression,
            "awaitPromise": await_promise,
            "returnByValue": True
        })
        val = res.get("result", {}).get("value")
        return val

    async def capture_screenshot(self, filepath):
        res = await self.send_cmd("Page.captureScreenshot", {"format": "png"})
        img_bytes = base64.b64decode(res["data"])
        with open(filepath, "wb") as f:
            f.write(img_bytes)
        print(f"Captured screenshot: {filepath} ({len(img_bytes)} bytes)")

    async def set_text(self, text):
        await self.evaluate_js("""
        (() => {
            const ta = document.querySelector('textarea');
            if (ta) {
                ta.focus();
                ta.select();
                document.execCommand('selectAll', false, null);
            }
        })()
        """)
        await self.send_cmd("Input.insertText", {"text": text})
        await self.evaluate_js("""
        (() => {
            const ta = document.querySelector('textarea');
            if (ta) {
                ta.dispatchEvent(new Event('input', { bubbles: true }));
                ta.dispatchEvent(new Event('change', { bubbles: true }));
                ta.blur();
            }
        })()
        """)
        await asyncio.sleep(1.0)
        return True

    async def click_button(self, label):
        js = f"""
        (() => {{
            const btns = Array.from(document.querySelectorAll('button'));
            const target = btns.find(b => b.innerText.includes({json.dumps(label)}));
            if (target) {{
                target.click();
                return true;
            }}
            return false;
        }})()
        """
        return await self.evaluate_js(js)

    async def wait_for_status_idle(self, timeout=30):
        start = time.time()
        while time.time() - start < timeout:
            # Check if streamlit is running status widget or analyzing
            is_running = await self.evaluate_js("""
            (() => {
                const status = document.querySelector('[data-testid="stStatusWidget"]');
                const spinners = document.querySelectorAll('.stSpinner');
                return (status !== null || spinners.length > 0);
            })()
            """)
            if not is_running:
                await asyncio.sleep(0.5)
                return True
            await asyncio.sleep(0.5)
        return False


async def run_automation():
    # 1. Fetch tabs
    url = f"http://127.0.0.1:{PORT}/json"
    for _ in range(10):
        try:
            with urllib.request.urlopen(url) as resp:
                tabs = json.loads(resp.read().decode())
                break
        except Exception:
            await asyncio.sleep(1)
    else:
        raise RuntimeError("Could not connect to Chrome debugging port")

    # Find the target tab
    target_tab = None
    for tab in tabs:
        if "8501" in tab.get("url", ""):
            target_tab = tab
            break
    if not target_tab and tabs:
        target_tab = tabs[0]

    ws_url = target_tab["webSocketDebuggerUrl"]
    print(f"Connecting to CDP: {ws_url}")
    cdp = CDPClient(ws_url)
    await cdp.connect()
    await cdp.send_cmd("Page.enable")
    await cdp.send_cmd("Runtime.enable")

    # Set viewport
    await cdp.send_cmd("Emulation.setDeviceMetricsOverride", {
        "width": 1280,
        "height": 960,
        "deviceScaleFactor": 1,
        "mobile": False
    })

    # Wait for page to fully render Streamlit
    print("Waiting for Streamlit initial render...")
    for _ in range(20):
        has_title = await cdp.evaluate_js("document.body.innerText.includes('TruthPulse AI')")
        if has_title:
            break
        await asyncio.sleep(0.5)

    await asyncio.sleep(2)

    # 1. Screenshot 1: Empty Page
    print("\n--- 1. Capturing Empty Page ---")
    ss1_path = os.path.join(SCREENSHOTS_DIR, "01_empty_page.png")
    await cdp.capture_screenshot(ss1_path)

    # 2. Screenshot 5: Error State (Empty submission error)
    print("\n--- 5. Capturing Error State (Empty Submit) ---")
    await cdp.click_button("Check News")
    await asyncio.sleep(1.5)
    ss5_path = os.path.join(SCREENSHOTS_DIR, "05_error_state.png")
    await cdp.capture_screenshot(ss5_path)

    # 3. Screenshot 2: FAKE result
    print("\n--- 2. Capturing FAKE Result ---")
    fake_claim = "Doctors confirm that drinking lemon water with salt permanently cures all forms of cancer within 48 hours without chemotherapy"
    await cdp.set_text(fake_claim)
    cnt = await cdp.evaluate_js("document.querySelector('.char-counter') ? document.querySelector('.char-counter').innerText : ''")
    print(f"Char counter before check: {cnt}")
    await cdp.click_button("Check News")
    print("Waiting for analysis pipeline to complete...")
    for _ in range(40):
        has_card = await cdp.evaluate_js("document.querySelector('.hero-card') !== null")
        if has_card:
            badge = await cdp.evaluate_js("document.querySelector('.hero-card').innerText.slice(0, 80)")
            print(f"Found hero card: {badge}")
            break
        await asyncio.sleep(1)
    await asyncio.sleep(1.5)
    ss2_path = os.path.join(SCREENSHOTS_DIR, "02_fake_result.png")
    await cdp.capture_screenshot(ss2_path)

    # 4. Screenshot 3: REAL result
    print("\n--- 3. Capturing REAL Result ---")
    real_claim = "Federal Reserve cuts benchmark interest rate by 50 basis points in September 2024 meeting"
    await cdp.click_button("Clear")
    await asyncio.sleep(1.5)
    await cdp.set_text(real_claim)
    cnt = await cdp.evaluate_js("document.querySelector('.char-counter') ? document.querySelector('.char-counter').innerText : ''")
    print(f"Char counter before check: {cnt}")
    await cdp.click_button("Check News")
    print("Waiting for analysis pipeline to complete...")
    for _ in range(40):
        badge = await cdp.evaluate_js("document.querySelector('.hero-card') ? document.querySelector('.hero-card').innerText.slice(0, 80) : ''")
        if "VERIFIED REAL" in badge:
            print(f"Found hero card: {badge}")
            break
        await asyncio.sleep(1)
    await asyncio.sleep(1.5)
    ss3_path = os.path.join(SCREENSHOTS_DIR, "03_real_result.png")
    await cdp.capture_screenshot(ss3_path)

    # 5. Screenshot 4: UNCERTAIN result
    print("\n--- 4. Capturing UNCERTAIN Result ---")
    uncertain_claim = "The United Nations officially announced the immediate dissolution of national borders across all member states under Agenda 2030."
    await cdp.click_button("Clear")
    await asyncio.sleep(1.5)
    await cdp.set_text(uncertain_claim)
    cnt = await cdp.evaluate_js("document.querySelector('.char-counter') ? document.querySelector('.char-counter').innerText : ''")
    print(f"Char counter before check: {cnt}")
    await cdp.click_button("Check News")
    print("Waiting for analysis pipeline to complete...")
    for _ in range(40):
        badge = await cdp.evaluate_js("document.querySelector('.hero-card') ? document.querySelector('.hero-card').innerText.slice(0, 80) : ''")
        if "UNCERTAIN" in badge:
            print(f"Found hero card: {badge}")
            break
        await asyncio.sleep(1)
    await asyncio.sleep(1.5)
    ss4_path = os.path.join(SCREENSHOTS_DIR, "04_uncertain_result.png")
    await cdp.capture_screenshot(ss4_path)

    print("\nAll 5 screenshots captured successfully!")


def main():
    chrome_cmd = [
        CHROME_BIN,
        "--headless=new",
        f"--remote-debugging-port={PORT}",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        f"--user-data-dir={USER_DATA_DIR}",
        "http://localhost:8501"
    ]
    print(f"Launching Chrome: {' '.join(chrome_cmd)}")
    proc = subprocess.Popen(chrome_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2)
    try:
        asyncio.run(run_automation())
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    main()
