/**
 * Grading Result screen.
 *
 * Displays the extracted student answer text with keyword highlights and
 * a marks entry panel. Marks are recorded in the local grade ledger.
 */

import * as Clipboard from "expo-clipboard";
import * as Sharing from "expo-sharing";
import { useLocalSearchParams, router } from "expo-router";
import { useRef, useState } from "react";
import {
  Alert,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import Markdown from "react-native-markdown-display";
import { useGradingStore } from "../stores/gradingStore";

export default function GradingResultScreen() {
  const params = useLocalSearchParams();
  const { session, addEntry, clearSession } = useGradingStore();

  // In a real app the result would be stored in the Zustand store after
  // the API call in capture.tsx. For this scaffold we show placeholder data.
  const markdown =
    (params.markdown as string) ??
    "_No extraction result available. Return to capture and try again._";
  const studentLabel = session.studentLabel || "Unknown student";
  const subject = session.subject || "Unknown subject";

  const [marks, setMarks] = useState("");
  const [maxMarks, setMaxMarks] = useState("");
  const [saved, setSaved] = useState(false);

  async function copyToClipboard() {
    await Clipboard.setStringAsync(markdown);
    Alert.alert("Copied", "Extracted text copied to clipboard.");
  }

  async function shareText() {
    const available = await Sharing.isAvailableAsync();
    if (!available) {
      Alert.alert("Sharing not available");
      return;
    }
    // Share as a temporary text file
    const { FileSystem } = await import("expo-file-system");
    const path = `${FileSystem.cacheDirectory}extracted_${Date.now()}.md`;
    await FileSystem.writeAsStringAsync(path, markdown);
    await Sharing.shareAsync(path, { mimeType: "text/markdown" });
  }

  function saveMarks() {
    const awarded = parseFloat(marks);
    const possible = parseFloat(maxMarks);

    if (isNaN(awarded) || isNaN(possible)) {
      Alert.alert("Invalid marks", "Enter valid numbers for marks and max marks.");
      return;
    }
    if (awarded > possible) {
      Alert.alert("Invalid marks", "Awarded marks cannot exceed max marks.");
      return;
    }

    const percentage = possible > 0 ? (awarded / possible) * 100 : 0;
    const grade =
      percentage >= 90 ? "A+" :
      percentage >= 80 ? "A" :
      percentage >= 70 ? "B" :
      percentage >= 60 ? "C" :
      percentage >= 50 ? "D" : "F";

    addEntry({
      id: `${Date.now()}-${Math.random()}`,
      studentLabel,
      subject,
      marksAwarded: awarded,
      marksPossible: possible,
      percentage: Math.round(percentage * 10) / 10,
      grade,
      createdAt: new Date().toISOString(),
      markdown,
    });

    setSaved(true);
    Alert.alert("Saved", `${studentLabel} — ${awarded}/${possible} (${grade})`);
  }

  function gradeAnother() {
    clearSession();
    router.replace("/capture");
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Student / subject header */}
      <View style={styles.header}>
        <Text style={styles.studentName}>{studentLabel}</Text>
        <Text style={styles.subject}>{subject}</Text>
      </View>

      {/* Metadata chips */}
      <View style={styles.chips}>
        <View style={styles.chip}>
          <Text style={styles.chipText}>Local</Text>
        </View>
      </View>

      {/* Extracted markdown */}
      <View style={styles.markdownCard}>
        <Text style={styles.sectionTitle}>Extracted Text</Text>
        <Markdown style={{ body: { color: "#1e293b", fontSize: 14, lineHeight: 22 } }}>
          {markdown}
        </Markdown>
      </View>

      {/* Marks entry */}
      <View style={styles.marksCard}>
        <Text style={styles.sectionTitle}>Record Marks</Text>
        <View style={styles.marksRow}>
          <View style={styles.marksField}>
            <Text style={styles.marksLabel}>Awarded</Text>
            <TextInput
              style={styles.marksInput}
              keyboardType="decimal-pad"
              value={marks}
              onChangeText={setMarks}
              placeholder="0"
            />
          </View>
          <Text style={styles.marksSlash}>/</Text>
          <View style={styles.marksField}>
            <Text style={styles.marksLabel}>Max</Text>
            <TextInput
              style={styles.marksInput}
              keyboardType="decimal-pad"
              value={maxMarks}
              onChangeText={setMaxMarks}
              placeholder="100"
            />
          </View>
        </View>
        <Pressable
          style={[styles.saveBtn, saved && styles.saveBtnSaved]}
          onPress={saveMarks}
          disabled={saved}
        >
          <Text style={styles.saveBtnText}>{saved ? "Saved ✓" : "Save to Ledger"}</Text>
        </Pressable>
      </View>

      {/* Actions */}
      <View style={styles.actionRow}>
        <Pressable style={styles.actionBtn} onPress={copyToClipboard}>
          <Text style={styles.actionBtnText}>Copy</Text>
        </Pressable>
        <Pressable style={styles.actionBtn} onPress={shareText}>
          <Text style={styles.actionBtnText}>Share</Text>
        </Pressable>
        <Pressable style={styles.actionBtn} onPress={() => router.push("/grade-ledger")}>
          <Text style={styles.actionBtnText}>Ledger</Text>
        </Pressable>
      </View>

      <Pressable style={styles.gradeAnotherBtn} onPress={gradeAnother}>
        <Text style={styles.gradeAnotherText}>Grade Another Student</Text>
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f8fafc" },
  content: { padding: 20, gap: 16 },
  header: {
    backgroundColor: "#1e3a5f",
    padding: 16,
    borderRadius: 12,
  },
  studentName: { fontSize: 18, fontWeight: "700", color: "#fff" },
  subject: { fontSize: 14, color: "#93c5fd", marginTop: 2 },
  chips: { flexDirection: "row", gap: 8 },
  chip: {
    backgroundColor: "#dbeafe",
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  chipText: { fontSize: 12, color: "#1e40af", fontWeight: "500" },
  markdownCard: {
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 16,
    shadowColor: "#000",
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  sectionTitle: { fontSize: 14, fontWeight: "600", color: "#475569", marginBottom: 10 },
  marksCard: {
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 16,
    shadowColor: "#000",
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  marksRow: { flexDirection: "row", alignItems: "center", gap: 12 },
  marksField: { flex: 1 },
  marksLabel: { fontSize: 12, color: "#64748b", marginBottom: 4 },
  marksInput: {
    borderWidth: 1,
    borderColor: "#cbd5e1",
    borderRadius: 8,
    padding: 10,
    fontSize: 20,
    fontWeight: "700",
    color: "#1e293b",
    textAlign: "center",
  },
  marksSlash: { fontSize: 24, color: "#64748b", marginTop: 20 },
  saveBtn: {
    backgroundColor: "#1e3a5f",
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: "center",
    marginTop: 12,
  },
  saveBtnSaved: { backgroundColor: "#16a34a" },
  saveBtnText: { color: "#fff", fontWeight: "600", fontSize: 15 },
  actionRow: { flexDirection: "row", gap: 10 },
  actionBtn: {
    flex: 1,
    backgroundColor: "#fff",
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#e2e8f0",
  },
  actionBtnText: { color: "#1e3a5f", fontWeight: "500", fontSize: 14 },
  gradeAnotherBtn: { alignItems: "center", paddingVertical: 12 },
  gradeAnotherText: { color: "#1e3a5f", fontSize: 15, fontWeight: "600" },
});
