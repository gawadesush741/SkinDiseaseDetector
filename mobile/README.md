# Dermalens AI - Mobile Application

Cross-platform native mobile application built with **React Native** and **Expo Camera** for realtime skin lesion classification.

---

## Quick Start (Run on Your Phone)

### 1. Prerequisites
- Node.js (version 18 or newer): [nodejs.org](https://nodejs.org)
- Free **Expo Go** app installed on your Android (Google Play) or iPhone (App Store).

### 2. Install Dependencies
```bash
cd mobile
npm install
```

### 3. Configure API Endpoint
In `mobile/App.js`:
- For testing on an Android emulator: `http://10.0.2.2:8000/predict`
- For testing on physical phones on the same Wi-Fi: `http://<YOUR_PC_LOCAL_IP>:8000/predict`
- For production: `https://<YOUR_RENDER_SUBDOMAIN>.onrender.com/predict`

### 4. Start Development Server
```bash
npx expo start
```
Scan the QR code displayed in your terminal using the **Expo Go** app on your phone.

---

## Build Standalone Android APK

To build a standalone installable `.apk` without Android Studio:
```bash
npx eas-cli build -p android --profile preview
```
This generates a direct `.apk` download link from the cloud!
