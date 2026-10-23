import React, { useState, useEffect, useRef } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  SafeAreaView,
  StatusBar,
  ActivityIndicator,
  ScrollView,
  Dimensions
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';

// Set this to your deployed backend URL (Render / Cloud / Local IP)
const API_URL = "http://10.0.2.2:8000/predict"; // Use your Render URL in production (e.g., https://your-app.onrender.com/predict)

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState('back');
  const [isScanning, setIsScanning] = useState(false);
  const [liveData, setLiveData] = useState(null);
  const [frozenReport, setFrozenReport] = useState(null);
  const [isBusy, setIsBusy] = useState(false);

  const cameraRef = useRef(null);
  const loopInterval = useRef(null);

  if (!permission) return <View style={styles.darkBg} />;

  if (!permission.granted) {
    return (
      <SafeAreaView style={[styles.darkBg, styles.center]}>
        <StatusBar barStyle="light-content" />
        <Text style={styles.promptTitle}>Camera Permission Required</Text>
        <Text style={styles.promptSub}>
          Dermalens requires camera access to perform real-time skin screening.
        </Text>
        <TouchableOpacity style={styles.primaryBtn} onPress={requestPermission}>
          <Text style={styles.primaryBtnText}>Grant Camera Access</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  const captureAndPredict = async () => {
    if (!cameraRef.current || isBusy) return;
    setIsBusy(true);

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.7,
        skipProcessing: true
      });

      const formData = new FormData();
      formData.append('file', {
        uri: photo.uri,
        name: 'scan.jpg',
        type: 'image/jpeg'
      });

      const res = await fetch(API_URL, {
        method: 'POST',
        body: formData,
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      if (res.ok) {
        const data = await res.json();
        setLiveData(data);
      }
    } catch (e) {
      console.log("Inference frame error:", e.message);
    } finally {
      setIsBusy(false);
    }
  };

  const toggleLiveScan = () => {
    if (isScanning) {
      clearInterval(loopInterval.current);
      loopInterval.current = null;
      setIsScanning(false);
    } else {
      setIsScanning(true);
      captureAndPredict();
      loopInterval.current = setInterval(captureAndPredict, 800);
    }
  };

  const freezeAndInspect = async () => {
    if (isScanning) {
      clearInterval(loopInterval.current);
      loopInterval.current = null;
      setIsScanning(false);
    }
    if (liveData) {
      setFrozenReport(liveData);
    } else {
      await captureAndPredict();
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#000" />

      {/* TOP HEADER */}
      <View style={styles.header}>
        <Text style={styles.brandTitle}>dermalens <Text style={styles.aiTag}>AI</Text></Text>
        <TouchableOpacity
          style={styles.flipBtn}
          onPress={() => setFacing(current => (current === 'back' ? 'front' : 'back'))}
        >
          <Text style={styles.flipBtnText}>🔄 Flip</Text>
        </TouchableOpacity>
      </View>

      {/* CAMERA VIEWPORT WITH HUD */}
      <View style={styles.cameraWrap}>
        <CameraView style={styles.camera} facing={facing} ref={cameraRef}>
          {/* Target Reticle */}
          <View style={styles.reticleWrap}>
            <View style={[styles.reticle, isScanning && styles.reticleActive]}>
              <View style={[styles.corner, styles.cornerTL]} />
              <View style={[styles.corner, styles.cornerTR]} />
              <View style={[styles.corner, styles.cornerBL]} />
              <View style={[styles.corner, styles.cornerBR]} />
            </View>

            <Text style={styles.hudStatus}>
              {isScanning ? "● LIVE REALTIME SCANNING..." : "ALIGN LESION IN RETICLE"}
            </Text>
          </View>

          {/* Floating Diagnostic Pill */}
          {liveData && (
            <View style={[
              styles.liveBadge,
              ["Melanoma, NOS", "Basal cell carcinoma"].includes(liveData.prediction)
                ? styles.badgeUrgent
                : styles.badgeBenign
            ]}>
              <Text style={styles.badgeCondition}>{liveData.prediction}</Text>
              <Text style={styles.badgeConfidence}>{liveData.confidence}% Confidence</Text>
            </View>
          )}
        </CameraView>
      </View>

      {/* ACTION CONTROLS */}
      <View style={styles.controlsBar}>
        <TouchableOpacity
          style={[styles.scanBtn, isScanning && styles.scanBtnActive]}
          onPress={toggleLiveScan}
        >
          <Text style={styles.scanBtnText}>
            {isScanning ? "⏸ PAUSE SCAN" : "📡 START LIVE SCAN"}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.freezeBtn} onPress={freezeAndInspect}>
          <Text style={styles.freezeBtnText}>🎯 FREEZE & REPORT</Text>
        </TouchableOpacity>
      </View>

      {/* DETAILED CLINICAL REPORT MODAL */}
      {frozenReport && (
        <View style={styles.reportOverlay}>
          <ScrollView contentContainerStyle={styles.reportScroll}>
            <View style={styles.reportHeader}>
              <Text style={styles.reportTitle}>Diagnostic Screening Summary</Text>
              <TouchableOpacity onPress={() => setFrozenReport(null)}>
                <Text style={styles.closeBtnText}>✕ Close</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.primaryResultBox}>
              <Text style={styles.resultLabel}>Primary AI Classification</Text>
              <Text style={styles.resultDisease}>{frozenReport.prediction}</Text>
              <Text style={styles.resultConf}>Confidence: {frozenReport.confidence}%</Text>
            </View>

            {frozenReport.top_predictions && (
              <View style={styles.diffBox}>
                <Text style={styles.diffHeader}>Differential Probabilities (Top-3):</Text>
                {frozenReport.top_predictions.map((p, idx) => (
                  <View key={idx} style={styles.diffRow}>
                    <Text style={styles.diffDisease}>{p.disease}</Text>
                    <Text style={styles.diffPct}>{p.confidence}%</Text>
                  </View>
                ))}
              </View>
            )}

            <View style={styles.detailSection}>
              <Text style={styles.detailHeader}>Clinical Description</Text>
              <Text style={styles.detailText}>{frozenReport.description}</Text>
            </View>

            <View style={styles.warningBox}>
              <Text style={styles.warningTitle}>⚠️ Medical Disclaimer</Text>
              <Text style={styles.warningText}>{frozenReport.medical_attention}</Text>
            </View>
          </ScrollView>
        </View>
      )}
    </SafeAreaView>
  );
}

const { width } = Dimensions.get('window');
const reticleSize = width * 0.65;

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0a100e' },
  darkBg: { flex: 1, backgroundColor: '#0a100e' },
  center: { justifyContent: 'center', alignItems: 'center', padding: 24 },
  promptTitle: { color: '#ffffff', fontSize: 20, fontWeight: '700', marginBottom: 8 },
  promptSub: { color: '#94a3b8', fontSize: 14, textAlign: 'center', marginBottom: 24 },
  primaryBtn: { backgroundColor: '#176a58', paddingVertical: 14, paddingHorizontal: 28, borderRadius: 999 },
  primaryBtnText: { color: '#ffffff', fontWeight: '700', fontSize: 15 },
  header: { height: 56, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 20 },
  brandTitle: { color: '#ffffff', fontSize: 20, fontWeight: '700' },
  aiTag: { color: '#34d399', fontSize: 13 },
  flipBtn: { paddingVertical: 6, paddingHorizontal: 12, backgroundColor: 'rgba(255,255,255,0.12)', borderRadius: 20 },
  flipBtnText: { color: '#ffffff', fontSize: 13, fontWeight: '600' },
  cameraWrap: { flex: 1, overflow: 'hidden', borderTopLeftRadius: 16, borderTopRightRadius: 16 },
  camera: { flex: 1, justifyContent: 'space-between', padding: 20 },
  reticleWrap: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  reticle: { width: reticleSize, height: reticleSize, borderRadius: 16, borderStyle: 'dashed', borderWidth: 1, borderColor: 'rgba(255,255,255,0.3)', position: 'relative' },
  reticleActive: { borderColor: '#34d399', borderWidth: 1.5 },
  corner: { position: 'absolute', width: 24, height: 24, borderColor: '#34d399', borderStyle: 'solid' },
  cornerTL: { top: -2, left: -2, borderTopWidth: 3, borderLeftWidth: 3, borderTopLeftRadius: 8 },
  cornerTR: { top: -2, right: -2, borderTopWidth: 3, borderRightWidth: 3, borderTopRightRadius: 8 },
  cornerBL: { bottom: -2, left: -2, borderBottomWidth: 3, borderLeftWidth: 3, borderBottomLeftRadius: 8 },
  cornerBR: { bottom: -2, right: -2, borderBottomWidth: 3, borderRightWidth: 3, borderBottomRightRadius: 8 },
  hudStatus: { color: '#ffffff', marginTop: 14, fontSize: 12, fontWeight: '700', letterSpacing: 0.8 },
  liveBadge: { alignSelf: 'center', paddingVertical: 10, paddingHorizontal: 20, borderRadius: 999, backgroundColor: 'rgba(18, 77, 66, 0.9)', alignItems: 'center' },
  badgeUrgent: { backgroundColor: 'rgba(185, 28, 28, 0.9)' },
  badgeBenign: { backgroundColor: 'rgba(18, 77, 66, 0.9)' },
  badgeCondition: { color: '#ffffff', fontSize: 16, fontWeight: '700' },
  badgeConfidence: { color: '#a7f3d0', fontSize: 12, marginTop: 2 },
  controlsBar: { padding: 18, flexDirection: 'row', gap: 12, backgroundColor: '#0a100e' },
  scanBtn: { flex: 1, paddingVertical: 14, backgroundColor: '#176a58', borderRadius: 10, alignItems: 'center' },
  scanBtnActive: { backgroundColor: '#eab308' },
  scanBtnText: { color: '#ffffff', fontWeight: '700', fontSize: 13 },
  freezeBtn: { flex: 1, paddingVertical: 14, backgroundColor: '#047857', borderRadius: 10, alignItems: 'center' },
  freezeBtnText: { color: '#ffffff', fontWeight: '700', fontSize: 13 },
  reportOverlay: { position: 'absolute', inset: 0, backgroundColor: 'rgba(0,0,0,0.85)', zIndex: 100 },
  reportScroll: { padding: 24 },
  reportHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 },
  reportTitle: { color: '#ffffff', fontSize: 18, fontWeight: '700' },
  closeBtnText: { color: '#94a3b8', fontSize: 16, fontWeight: '700' },
  primaryResultBox: { padding: 20, backgroundColor: '#132e27', borderRadius: 12, marginBottom: 16 },
  resultLabel: { color: '#6ee7b7', fontSize: 11, fontWeight: '700', textTransform: 'uppercase' },
  resultDisease: { color: '#ffffff', fontSize: 22, fontWeight: '700', marginTop: 4 },
  resultConf: { color: '#a7f3d0', fontSize: 13, marginTop: 2 },
  diffBox: { padding: 16, backgroundColor: '#16221f', borderRadius: 10, marginBottom: 16 },
  diffHeader: { color: '#ffffff', fontSize: 13, fontWeight: '700', marginBottom: 8 },
  diffRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 4 },
  diffDisease: { color: '#cbd5e1', fontSize: 13 },
  diffPct: { color: '#34d399', fontSize: 13, fontWeight: '700' },
  detailSection: { marginBottom: 16 },
  detailHeader: { color: '#ffffff', fontSize: 14, fontWeight: '700', marginBottom: 6 },
  detailText: { color: '#cbd5e1', fontSize: 13, lineHeight: 20 },
  warningBox: { padding: 16, backgroundColor: '#2d1512', borderRadius: 10, borderWidth: 1, borderColor: '#7f1d1d' },
  warningTitle: { color: '#fca5a5', fontSize: 13, fontWeight: '700', marginBottom: 4 },
  warningText: { color: '#fecaca', fontSize: 12, lineHeight: 18 }
});
