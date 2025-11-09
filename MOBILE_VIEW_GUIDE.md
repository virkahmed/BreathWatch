# How to View Mobile View

## Option 1: Browser DevTools (Recommended)

1. **Open the app in your browser** (usually `http://localhost:8081` or similar)

2. **Open Developer Tools:**
   - **Windows/Linux:** Press `F12` or `Ctrl+Shift+I`
   - **Mac:** Press `Cmd+Option+I`

3. **Toggle Device Toolbar:**
   - Click the device icon in the toolbar (looks like a phone/tablet)
   - OR press `Ctrl+Shift+M` (Windows/Linux) or `Cmd+Shift+M` (Mac)

4. **Select a device:**
   - Click the device dropdown at the top
   - Choose a device like:
     - iPhone 12 Pro
     - iPhone 14 Pro
     - Samsung Galaxy S20
     - Or set a custom size (e.g., 375px width for iPhone)

5. **Refresh if needed:**
   - Press `F5` or `Cmd+R` to refresh the page

## Option 2: Resize Browser Window

Simply drag your browser window to make it narrow (around 375-414px wide for mobile size).

## Option 3: Test on Real Device

If you're running Expo:

1. Start the dev server: `npm start`
2. Scan the QR code with:
   - **iOS:** Camera app or Expo Go app
   - **Android:** Expo Go app
3. The app will open on your phone

## Option 4: Use Browser Responsive Design Mode

1. Open DevTools (`F12` or `Cmd+Option+I`)
2. Click the "Toggle device toolbar" button
3. Select "Responsive" mode
4. Set width to 375px (iPhone) or 414px (iPhone Plus)

---

**Tip:** The action bar at the bottom is optimized for mobile and will show smaller icons and text on narrow screens!

