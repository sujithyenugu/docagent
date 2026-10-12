/**
 * Home screen — shows quick-start buttons and session summary.
 */

import { useQuery } from "@tanstack/react-query";
import { router } from "expo-router";
import {
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { getHealth } from "../services/api";
import { useGradingStore } from "../stores/gradingStore";

export default function HomeScreen() {
  const ledger = useGradingStore((s) => s.ledger);

  const { data: health, isError } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30_000,
  });

  const backendLabel = health?.local_model_available
    ? "Local (on-device)"
    : health?.cloud_api_configured
    ? "Cloud (Landing AI)"
    : "Not configured";

  const backendColor = health?.local_model_available
    ? "#16a34a"
    : health?.cloud_api_configured
    ? "#ca8a04"
    : "#dc2626";

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Backend status */}
      <View style={styles.statusRow}>
        <View style={[styles.dot, { backgroundColor: backendColor }]} />
        <Text style={styles.statusText}>
          Backend: {isError ? "Unreachable" : backendLabel}
        </Text>
      </View>

      <Text style={styles.heading}>Teacher Grading</Text>
      <Text style={styles.subtitle}>
        Photograph student answer sheets, extract text, and record marks.
      </Text>

      {/* Primary CTA */}
      <Pressable
        style={[styles.primaryButton]}
        onPress={() => router.push("/capture")}
      >
        <Text style={styles.primaryButtonText}>Start Grading Session</Text>
      </Pressable>

      {/* Secondary actions */}
      <View style={styles.rowButtons}>
        <Pressable
          style={styles.secondaryButton}
          onPress={() => router.push("/grade-ledger")}
        >
          <Text style={styles.secondaryButtonText}>
            Grade Ledger ({ledger.length})
          </Text>
        </Pressable>
        <Pressable
          style={styles.secondaryButton}
          onPress={() => router.push("/mark-sheet-review")}
        >
          <Text style={styles.secondaryButtonText}>Fill Mark Sheet</Text>
        </Pressable>
      </View>

      <Pressable
        style={styles.settingsLink}
        onPress={() => router.push("/settings")}
      >
        <Text style={styles.settingsLinkText}>Settings</Text>
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f8fafc" },
  content: { padding: 24, gap: 16 },
  statusRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingVertical: 8,
    paddingHorizontal: 12,
    backgroundColor: "#fff",
    borderRadius: 8,
    shadowColor: "#000",
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  dot: { width: 10, height: 10, borderRadius: 5 },
  statusText: { fontSize: 13, color: "#475569" },
  heading: {
    fontSize: 28,
    fontWeight: "700",
    color: "#1e3a5f",
    marginTop: 16,
  },
  subtitle: { fontSize: 15, color: "#64748b", lineHeight: 22 },
  primaryButton: {
    backgroundColor: "#1e3a5f",
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: "center",
    marginTop: 8,
  },
  primaryButtonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
  rowButtons: { flexDirection: "row", gap: 12 },
  secondaryButton: {
    flex: 1,
    backgroundColor: "#fff",
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#e2e8f0",
  },
  secondaryButtonText: { color: "#1e3a5f", fontSize: 14, fontWeight: "500" },
  settingsLink: { alignItems: "center", marginTop: 8 },
  settingsLinkText: { color: "#64748b", fontSize: 14 },
});
