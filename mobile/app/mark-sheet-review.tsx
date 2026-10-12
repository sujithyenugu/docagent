/**
 * Mark Sheet Review screen.
 *
 * Teachers select a grading form template (CBSE/ICSE mark sheet, progress
 * report, admit card verification), upload student documents, and review/
 * edit the auto-extracted fields before generating a PDF.
 */

import { useQuery } from "@tanstack/react-query";
import * as ImagePicker from "expo-image-picker";
import * as Print from "expo-print";
import * as Sharing from "expo-sharing";
import { useState } from "react";
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { fillForm, getFormTemplates } from "../services/api";
import { useGradingStore } from "../stores/gradingStore";

type Field = { key: string; label: string; value: string; required: boolean; confidence: "high" | "low" | "missing" };

export default function MarkSheetReviewScreen() {
  const { session } = useGradingStore();

  const { data: templates, isLoading: loadingTemplates } = useQuery({
    queryKey: ["formTemplates"],
    queryFn: getFormTemplates,
  });

  const gradingTemplates = templates?.filter((t) => t.use_case === "grading") ?? [];

  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null);
  const [fileUris, setFileUris] = useState<string[]>([]);
  const [fields, setFields] = useState<Field[]>([]);
  const [missingFields, setMissingFields] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [consentGiven, setConsentGiven] = useState(false);

  const selectedTemplate = templates?.find((t) => t.template_id === selectedTemplateId);

  async function pickDocument() {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.85,
    });
    if (!result.canceled && result.assets[0]) {
      setFileUris((prev) => [...prev, result.assets[0].uri]);
    }
  }

  async function runExtraction() {
    if (!selectedTemplateId || fileUris.length === 0) {
      Alert.alert("Missing input", "Select a template and upload at least one document.");
      return;
    }
    if (!consentGiven) {
      Alert.alert("Consent required", "Please confirm the consent statement below before processing.");
      return;
    }

    setLoading(true);
    try {
      const filenames = fileUris.map((u, i) => u.split("/").pop() ?? `doc_${i}.jpg`);
      const result = await fillForm(fileUris, filenames, selectedTemplateId, {
        useLocal: session.useLocal,
        apiKey: session.apiKey || undefined,
      });

      const allFields = [
        ...(selectedTemplate?.required_fields ?? []).map((k) => ({ key: k, required: true })),
        ...(selectedTemplate?.optional_fields ?? []).map((k) => ({ key: k, required: false })),
      ];

      const parsed: Field[] = allFields.map(({ key, required }) => {
        const val = (result.fields[key] ?? "") as string;
        const missing = result.missing_fields.includes(key);
        return {
          key,
          label: key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
          value: val,
          required,
          confidence: missing ? "missing" : val ? "high" : "low",
        };
      });

      setFields(parsed);
      setMissingFields(result.missing_fields);
    } catch (err: unknown) {
      Alert.alert("Extraction failed", err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  function updateField(key: string, value: string) {
    setFields((prev) =>
      prev.map((f) =>
        f.key === key
          ? { ...f, value, confidence: value ? "high" : "missing" }
          : f
      )
    );
  }

  async function generatePdf() {
    const rows = fields
      .map((f) => `<tr><td><b>${f.label}</b></td><td>${f.value || "—"}</td></tr>`)
      .join("");
    const html = `
      <html><body style="font-family:sans-serif;padding:24px">
        <h2>${selectedTemplateId?.replace(/_/g, " ").toUpperCase()}</h2>
        <table border="1" cellpadding="8" style="border-collapse:collapse;width:100%">
          ${rows}
        </table>
      </body></html>
    `;
    const { uri } = await Print.printToFileAsync({ html });
    const ok = await Sharing.isAvailableAsync();
    if (ok) await Sharing.shareAsync(uri, { mimeType: "application/pdf" });
  }

  const confidenceColor = (c: Field["confidence"]) =>
    c === "high" ? "#16a34a" : c === "low" ? "#ca8a04" : "#dc2626";

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.heading}>Mark Sheet Review</Text>

      {/* Template selector */}
      {loadingTemplates ? (
        <ActivityIndicator />
      ) : (
        <>
          <Text style={styles.label}>Select Template</Text>
          <View style={styles.templateList}>
            {gradingTemplates.map((t) => (
              <Pressable
                key={t.template_id}
                style={[
                  styles.templateChip,
                  selectedTemplateId === t.template_id && styles.templateChipSelected,
                ]}
                onPress={() => setSelectedTemplateId(t.template_id)}
              >
                <Text
                  style={[
                    styles.templateChipText,
                    selectedTemplateId === t.template_id && styles.templateChipTextSelected,
                  ]}
                >
                  {t.template_id.replace(/_/g, " ")}
                </Text>
              </Pressable>
            ))}
          </View>
        </>
      )}

      {/* Document upload */}
      <Pressable style={styles.uploadBtn} onPress={pickDocument}>
        <Text style={styles.uploadBtnText}>Add Document ({fileUris.length} added)</Text>
      </Pressable>

      {/* Consent */}
      <Pressable
        style={styles.consentRow}
        onPress={() => setConsentGiven((v) => !v)}
      >
        <View style={[styles.checkbox, consentGiven && styles.checkboxChecked]}>
          {consentGiven && <Text style={styles.checkmark}>✓</Text>}
        </View>
        <Text style={styles.consentText}>
          I confirm I have authority to process these student documents for educational purposes.
        </Text>
      </Pressable>

      {/* Extract button */}
      {loading ? (
        <ActivityIndicator size="large" color="#1e3a5f" />
      ) : (
        <Pressable style={styles.extractBtn} onPress={runExtraction}>
          <Text style={styles.extractBtnText}>Extract Fields</Text>
        </Pressable>
      )}

      {/* Missing fields warning */}
      {missingFields.length > 0 && (
        <View style={styles.warningBox}>
          <Text style={styles.warningText}>
            Missing required fields: {missingFields.join(", ")}
          </Text>
        </View>
      )}

      {/* Editable field list */}
      {fields.length > 0 && (
        <>
          <Text style={styles.sectionTitle}>Extracted Fields</Text>
          {fields.map((f) => (
            <View key={f.key} style={styles.fieldRow}>
              <View style={styles.fieldLabelRow}>
                <Text style={styles.fieldLabel}>{f.label}</Text>
                <View
                  style={[
                    styles.confidenceDot,
                    { backgroundColor: confidenceColor(f.confidence) },
                  ]}
                />
              </View>
              <TextInput
                style={[
                  styles.fieldInput,
                  f.confidence === "missing" && styles.fieldInputMissing,
                ]}
                value={f.value}
                onChangeText={(v) => updateField(f.key, v)}
                placeholder={f.required ? "Required" : "Optional"}
                placeholderTextColor="#94a3b8"
              />
            </View>
          ))}

          <Pressable style={styles.pdfBtn} onPress={generatePdf}>
            <Text style={styles.pdfBtnText}>Generate Grade Report PDF</Text>
          </Pressable>
        </>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f8fafc" },
  content: { padding: 20, gap: 14 },
  heading: { fontSize: 22, fontWeight: "700", color: "#1e3a5f" },
  label: { fontSize: 14, fontWeight: "600", color: "#334155" },
  templateList: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  templateChip: {
    borderWidth: 1,
    borderColor: "#94a3b8",
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  templateChipSelected: { backgroundColor: "#1e3a5f", borderColor: "#1e3a5f" },
  templateChipText: { fontSize: 13, color: "#475569" },
  templateChipTextSelected: { color: "#fff" },
  uploadBtn: {
    borderWidth: 1,
    borderColor: "#1e3a5f",
    borderStyle: "dashed",
    borderRadius: 10,
    padding: 14,
    alignItems: "center",
  },
  uploadBtnText: { color: "#1e3a5f", fontWeight: "500" },
  consentRow: { flexDirection: "row", gap: 10, alignItems: "flex-start" },
  checkbox: {
    width: 22,
    height: 22,
    borderWidth: 2,
    borderColor: "#94a3b8",
    borderRadius: 4,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 1,
  },
  checkboxChecked: { backgroundColor: "#1e3a5f", borderColor: "#1e3a5f" },
  checkmark: { color: "#fff", fontSize: 14, fontWeight: "700" },
  consentText: { flex: 1, fontSize: 13, color: "#475569", lineHeight: 20 },
  extractBtn: {
    backgroundColor: "#1e3a5f",
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
  },
  extractBtnText: { color: "#fff", fontSize: 15, fontWeight: "600" },
  warningBox: {
    backgroundColor: "#fef3c7",
    borderRadius: 8,
    padding: 12,
    borderLeftWidth: 4,
    borderLeftColor: "#f59e0b",
  },
  warningText: { color: "#92400e", fontSize: 13 },
  sectionTitle: { fontSize: 15, fontWeight: "600", color: "#1e3a5f" },
  fieldRow: { gap: 4 },
  fieldLabelRow: { flexDirection: "row", alignItems: "center", gap: 6 },
  fieldLabel: { fontSize: 13, fontWeight: "500", color: "#475569" },
  confidenceDot: { width: 8, height: 8, borderRadius: 4 },
  fieldInput: {
    borderWidth: 1,
    borderColor: "#cbd5e1",
    borderRadius: 8,
    padding: 10,
    fontSize: 14,
    color: "#1e293b",
    backgroundColor: "#fff",
  },
  fieldInputMissing: { borderColor: "#fca5a5", backgroundColor: "#fff5f5" },
  pdfBtn: {
    backgroundColor: "#7c3aed",
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
    marginTop: 8,
  },
  pdfBtnText: { color: "#fff", fontSize: 15, fontWeight: "600" },
});
