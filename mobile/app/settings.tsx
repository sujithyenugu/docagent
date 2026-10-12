/**
 * Settings screen — backend selection, API key, and privacy consent.
 */

import { useState } from "react";
import {
  Alert,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from "react-native";
import { useGradingStore } from "../stores/gradingStore";

export default function SettingsScreen() {
  const { session, setUseLocal, setApiKey } = useGradingStore();
  const [keyInput, setKeyInput] = useState(session.apiKey);

  function handleToggleBackend(value: boolean) {
    if (!value && !keyInput) {
      Alert.alert(
        "Cloud backend requires an API key",
        "Enter your Landing AI API key to use cloud extraction.",
        [{ text: "OK" }]
      );
    }
    setUseLocal(value);
  }

  function saveKey() {
    setApiKey(keyInput.trim());
    Alert.alert("Saved", "API key updated.");
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.heading}>Settings</Text>

      {/* Backend toggle */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Extraction Backend</Text>
        <View style={styles.toggleRow}>
          <Text style={styles.toggleLabel}>
            {session.useLocal ? "Local (on-device, Apple MLX)" : "Cloud (Landing AI)"}
          </Text>
          <Switch
            value={session.useLocal}
            onValueChange={handleToggleBackend}
            trackColor={{ false: "#ca8a04", true: "#16a34a" }}
            thumbColor="#fff"
          />
        </View>
        <Text style={styles.hint}>
          {session.useLocal
            ? "Local mode: no data leaves the device. Runs entirely on the Mac — no setup needed."
            : "Cloud mode: images sent to Landing AI servers. Requires API key."}
        </Text>
      </View>

      {/* API key */}
      {!session.useLocal && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Landing AI API Key</Text>
          <TextInput
            style={styles.input}
            value={keyInput}
            onChangeText={setKeyInput}
            placeholder="Enter your Landing AI API key"
            secureTextEntry
            placeholderTextColor="#94a3b8"
          />
          <Pressable style={styles.saveBtn} onPress={saveKey}>
            <Text style={styles.saveBtnText}>Save Key</Text>
          </Pressable>
        </View>
      )}

      {/* Privacy notice */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Privacy Notice</Text>
        <Text style={styles.privacyText}>
          DocumentAgent processes photographs of student answer sheets and
          assignments to assist with grading.{"\n\n"}
          In <Text style={{ fontWeight: "700" }}>local mode</Text>, all
          processing happens on this device or your school's server — no data is
          sent externally.{"\n\n"}
          In <Text style={{ fontWeight: "700" }}>cloud mode</Text>, images are
          sent to Landing AI's servers for processing. Ensure you have
          appropriate consent from students or guardians before using this mode
          with identifiable documents.{"\n\n"}
          No extracted text or images are stored by the API. Marks are saved
          locally on this device only.
        </Text>
      </View>

      <Text style={styles.version}>DocumentAgent v1.0.0</Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f8fafc" },
  content: { padding: 20, gap: 16 },
  heading: { fontSize: 22, fontWeight: "700", color: "#1e3a5f" },
  card: {
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 16,
    gap: 10,
    shadowColor: "#000",
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  cardTitle: { fontSize: 15, fontWeight: "600", color: "#1e3a5f" },
  toggleRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  toggleLabel: { fontSize: 14, color: "#334155", flex: 1 },
  hint: { fontSize: 12, color: "#64748b", lineHeight: 18 },
  input: {
    borderWidth: 1,
    borderColor: "#cbd5e1",
    borderRadius: 8,
    padding: 10,
    fontSize: 14,
    color: "#1e293b",
  },
  saveBtn: {
    backgroundColor: "#1e3a5f",
    paddingVertical: 10,
    borderRadius: 8,
    alignItems: "center",
  },
  saveBtnText: { color: "#fff", fontWeight: "600" },
  privacyText: { fontSize: 13, color: "#475569", lineHeight: 20 },
  version: { textAlign: "center", color: "#94a3b8", fontSize: 12 },
});
