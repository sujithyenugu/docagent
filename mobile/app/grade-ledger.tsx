/**
 * Grade Ledger screen.
 *
 * Shows all graded entries for the current session. Entries persist in the
 * Zustand store (and SQLite in a full implementation).
 */

import * as Sharing from "expo-sharing";
import { router } from "expo-router";
import {
  Alert,
  FlatList,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { useGradingStore, GradingEntry } from "../stores/gradingStore";

function GradeTag({ grade }: { grade: string }) {
  const colors: Record<string, string> = {
    "A+": "#166534",
    A: "#15803d",
    B: "#1d4ed8",
    C: "#ca8a04",
    D: "#ea580c",
    F: "#dc2626",
  };
  const bg = colors[grade] ?? "#475569";
  return (
    <View style={[styles.gradeTag, { backgroundColor: bg }]}>
      <Text style={styles.gradeTagText}>{grade}</Text>
    </View>
  );
}

function EntryCard({ entry, onRemove }: { entry: GradingEntry; onRemove: () => void }) {
  const date = new Date(entry.createdAt);
  const dateStr = date.toLocaleDateString("en-IN", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });

  return (
    <View style={styles.card}>
      <View style={styles.cardHeader}>
        <View style={{ flex: 1 }}>
          <Text style={styles.cardName}>{entry.studentLabel}</Text>
          <Text style={styles.cardSubject}>{entry.subject}</Text>
        </View>
        <GradeTag grade={entry.grade} />
      </View>
      <View style={styles.cardBody}>
        <Text style={styles.cardMarks}>
          {entry.marksAwarded}/{entry.marksPossible} ({entry.percentage}%)
        </Text>
        <Text style={styles.cardDate}>{dateStr}</Text>
      </View>
      <Pressable style={styles.removeBtn} onPress={onRemove}>
        <Text style={styles.removeBtnText}>Remove</Text>
      </Pressable>
    </View>
  );
}

export default function GradeLedgerScreen() {
  const { ledger, removeEntry } = useGradingStore();

  const totalStudents = ledger.length;
  const avgPercentage =
    ledger.length > 0
      ? Math.round(
          (ledger.reduce((sum, e) => sum + e.percentage, 0) / ledger.length) * 10
        ) / 10
      : null;

  async function exportLedger() {
    const csv = [
      "Student,Subject,Marks Awarded,Marks Possible,Percentage,Grade,Date",
      ...ledger.map((e) =>
        [
          `"${e.studentLabel}"`,
          `"${e.subject}"`,
          e.marksAwarded,
          e.marksPossible,
          e.percentage,
          e.grade,
          `"${new Date(e.createdAt).toLocaleDateString("en-IN")}"`,
        ].join(",")
      ),
    ].join("\n");

    const { FileSystem } = await import("expo-file-system");
    const path = `${FileSystem.cacheDirectory}grade_ledger_${Date.now()}.csv`;
    await FileSystem.writeAsStringAsync(path, csv);

    const ok = await Sharing.isAvailableAsync();
    if (ok) await Sharing.shareAsync(path, { mimeType: "text/csv" });
    else Alert.alert("Sharing not available on this device.");
  }

  function confirmRemove(id: string) {
    Alert.alert("Remove entry", "Delete this grade record?", [
      { text: "Cancel", style: "cancel" },
      { text: "Delete", style: "destructive", onPress: () => removeEntry(id) },
    ]);
  }

  if (ledger.length === 0) {
    return (
      <View style={styles.empty}>
        <Text style={styles.emptyText}>No graded entries yet.</Text>
        <Pressable style={styles.startBtn} onPress={() => router.push("/capture")}>
          <Text style={styles.startBtnText}>Start Grading</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <FlatList
      style={styles.container}
      contentContainerStyle={styles.content}
      data={ledger}
      keyExtractor={(item) => item.id}
      ListHeaderComponent={
        <View>
          {/* Summary */}
          <View style={styles.summaryRow}>
            <View style={styles.summaryCard}>
              <Text style={styles.summaryNum}>{totalStudents}</Text>
              <Text style={styles.summaryLabel}>Students</Text>
            </View>
            {avgPercentage !== null && (
              <View style={styles.summaryCard}>
                <Text style={styles.summaryNum}>{avgPercentage}%</Text>
                <Text style={styles.summaryLabel}>Class Avg</Text>
              </View>
            )}
          </View>
          <Pressable style={styles.exportBtn} onPress={exportLedger}>
            <Text style={styles.exportBtnText}>Export CSV</Text>
          </Pressable>
          <Text style={styles.sectionTitle}>All Entries</Text>
        </View>
      }
      renderItem={({ item }) => (
        <EntryCard entry={item} onRemove={() => confirmRemove(item.id)} />
      )}
      ItemSeparatorComponent={() => <View style={{ height: 10 }} />}
    />
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f8fafc" },
  content: { padding: 20, gap: 12 },
  empty: { flex: 1, justifyContent: "center", alignItems: "center", gap: 16 },
  emptyText: { fontSize: 16, color: "#94a3b8" },
  startBtn: {
    backgroundColor: "#1e3a5f",
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 10,
  },
  startBtnText: { color: "#fff", fontWeight: "600" },
  summaryRow: { flexDirection: "row", gap: 12, marginBottom: 12 },
  summaryCard: {
    flex: 1,
    backgroundColor: "#1e3a5f",
    borderRadius: 10,
    padding: 16,
    alignItems: "center",
  },
  summaryNum: { fontSize: 28, fontWeight: "700", color: "#fff" },
  summaryLabel: { fontSize: 12, color: "#93c5fd", marginTop: 2 },
  exportBtn: {
    borderWidth: 1,
    borderColor: "#1e3a5f",
    borderRadius: 8,
    padding: 10,
    alignItems: "center",
    marginBottom: 12,
  },
  exportBtnText: { color: "#1e3a5f", fontWeight: "500" },
  sectionTitle: { fontSize: 15, fontWeight: "600", color: "#1e3a5f", marginBottom: 4 },
  card: {
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 14,
    shadowColor: "#000",
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  cardHeader: { flexDirection: "row", alignItems: "flex-start" },
  cardName: { fontSize: 16, fontWeight: "600", color: "#1e293b" },
  cardSubject: { fontSize: 13, color: "#64748b", marginTop: 2 },
  gradeTag: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
  },
  gradeTagText: { color: "#fff", fontWeight: "700", fontSize: 15 },
  cardBody: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 8,
  },
  cardMarks: { fontSize: 15, fontWeight: "500", color: "#1e293b" },
  cardDate: { fontSize: 12, color: "#94a3b8" },
  removeBtn: { marginTop: 8, alignSelf: "flex-end" },
  removeBtnText: { color: "#dc2626", fontSize: 12 },
});
