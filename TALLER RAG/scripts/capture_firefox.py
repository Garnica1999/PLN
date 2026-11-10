"""Captura una página web con Firefox headless, con login opcional en Langfuse.

Uso:
    python scripts/capture_firefox.py --url URL --output reports/figuras/x.png \
        --wait-text "baseline" --email carlos@taller.local --password ...
"""
from __future__ import annotations

import argparse
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait


def sign_in(driver, email: str, password: str) -> None:
    """Completa el formulario de Langfuse si la página redirige al login."""
    if "/auth/sign-in" not in driver.current_url:
        return
    wait = WebDriverWait(driver, 40)
    wait.until(lambda page: page.find_elements(By.NAME, "email"))
    driver.find_element(By.NAME, "email").send_keys(email)
    driver.find_element(By.NAME, "password").send_keys(password)
    driver.find_element(By.XPATH, "//button[normalize-space()='Sign in']").click()
    wait.until(lambda page: "/auth/sign-in" not in page.current_url)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait-text", default="")
    parser.add_argument("--email", default="")
    parser.add_argument("--password", default="")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1100)
    parser.add_argument("--settle", type=float, default=3.0)
    args = parser.parse_args()

    options = Options()
    options.add_argument("-headless")
    options.set_preference("network.proxy.type", 0)
    driver = webdriver.Firefox(options=options)
    driver.set_window_size(args.width, args.height)
    driver.get(args.url)
    time.sleep(2)
    if args.email:
        sign_in(driver, args.email, args.password)
        driver.get(args.url)
    if args.wait_text:
        WebDriverWait(driver, 60).until(
            lambda page: args.wait_text.lower() in page.find_element(By.TAG_NAME, "body").text.lower()
        )
    time.sleep(args.settle)
    driver.save_screenshot(args.output)
    print("body:", driver.find_element(By.TAG_NAME, "body").text[:600].replace("\n", " | "))
    print("saved:", args.output)
    driver.quit()


if __name__ == "__main__":
    main()
