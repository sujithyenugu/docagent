/**
 * Assignment Capture screen.
 *
 * Teachers photograph student answer sheets page by page. Each page is
 * shown in a preview strip with a remove button. A student label (name
 * or roll number) is tagged to the capture before submitting.
 */

import * as DocumentPicker from "expo-document-picker";
import * as ImagePicker from "expo-image-picker";
import { router } from "expo-router";
import { useState } from "react";
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { gradeExtract } from "../services/api";
import { useGradingStore } from "../stores/gradingStore";

const MAX_PAGES = 5;

export default function CaptureScreen() {
  const {
    session,
    setStudentLabel,
    setSubject,
    addPage,
    removePage,
    clearSession,
  } = useGradingStore();

  const [loading, setLoading] = useState(false);

  async function pickFromCamera() {
    if (session.pages.length >= MAX_PAGES) {
      Alert.alert("Limit reached", `Maximum ${MAX_PAGES} pages per submission.`);
      return;
    }
    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.85,
      allowsEditing: false,
    });
    if (!result.canceled && result.assets[0]) {
      addPage({
        uri: result.assets[0].uri,
        pageNumber: session.pages.length + 1,
      });
    }
  }

  async function pickFromGallery() {
    if (session.pages.length >= MAX_PAGES) {
      Alert.alert("Limit reached", `Maximum ${MAX_PAGES} pages per submission.`);
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.85,
      allowsEditing: false,
    });
    if (!result.canceled && result.assets[0]) {
      addPage({
        uri: result.assets[0].uri,
        pageNumber: session.pages.length + 1,
      });
    }
  }

  async function pickPdf() {
    const result = await DocumentPicker.getDocumentAsync({ type: "application/pdf" });
    if (!result.canceled && result.assets[0]) {
      addPage({
        uri: result.assets[0].uri,
        pageNumber: session.pages.length + 1,
      });
    }
  }

  async function submit() {
    if (session.pages.length === 0) {
      Alert.alert("No pages", "Please capture at least one page.");
      return;
    }
    setLoading(true);
    try {
      const firstPage = session.pages[0];
      const filename = firstPage.uri.split("/").pop() ?? "page_1.jpg";

      const result = await gradeExtract(firstPage.uri, filename, {
        useLocal: session.useLocal,
        apiKey: session.apiKey || undefined,
        studentLabel: session.studentLabel || undefined,
      });

      // Navigate to result screen, pass data via global store
      router.push({ pathname: "/grading-result", params: { requestId: result.request_id } });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      Alert.alert("Extraction failed", msg);
    } finally {
      setLoading(false);
    }
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Student label */}
      <Text style={styles.label}>Student Name / Roll Number</Text>
      <TextInput
        style={styles.input}
        value={session.studentLabel}
        onChangeText={setStudentLabel}
        placeholder="e.g. Priya Sharma / Roll 42"
        placeholderTextColor="#94a3b8"
      />

      {/* Subject */}
      <Text style={styles.label}>Subject</Text>
      <TextInput
        style={styles.input}
        value={session.subject}
        onChangeText={setSubject}
        placeholder="e.g. Science, Mathematics"
        placeholderTextColor="#94a3b8"
      />

      {/* Capture buttons */}
      <View style={styles.captureRow}>
        <Pressable style={styles.captureBtn} onPress={pickFromCamera}>
          <Text style={styles.captureBtnText}>Camera</Text>
        </Pressable>
        <Pressable style={styles.captureBtn} onPress={pickFromGallery}>
          <Text style={styles.captureBtnText}>Gallery</Text>
        </Pressable>
        <Pressable style={styles.captureBtn} onPress={pickPdf}>
          <Text style={styles.captureBtnText}>PDF</Text>
        </Pressable>
      </View>

      {/* Page count */}
      {session.pages.length > 0 && (
        <Text style={styles.pageCount}>
          {session.pages.length} / {MAX_PAGES} pages captured
        </Text>
      )}

      {/* Page preview strip */}
      {session.pages.length > 0 && (
        <FlatList
          data={session.pages}
          horizontal
          keyExtractor={(item) => String(item.pageNumber)}
          contentContainerStyle={styles.previewStrip}
          renderItem={({ item }) => (
            <View style={styles.thumbContainer}>
              <Image source={{ uri: item.uri }} style={styles.thumb} />
              <Text style={styles.thumbLabel}>Page {item.pageNumber}</Text>
              <Pressable
                style={styles.removeBtn}
                onPress={() => removePage(item.pageNumber)}
              >
                <Text style={styles.removeBtnText}>✕</Text>
              </Pressable>
            </View>
          )}
        />
      )}

      {/* Submit */}
      {loading ? (
        <ActivityIndicator size="large" color="#1e3a5f" style={{ marginTop: 24 }} />
      ) : (
        <Pressable
          style={[
            styles.submitBtn,
            session.pages.length === 0 && styles.submitBtnDisabled,
          ]}
          onPress={submit}
          disabled={session.pages.length === 0}
        >
          <Text style={styles.submitBtnText}>Extract & Grade</Text>
        </Pressable>
      )}

      <Pressable
        style={styles.clearBtn}
        onPress={() => {
          clearSession();
        }}
      >
        <Text style={styles.clearBtnText}>Clear Session</Text>
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f8fafc" },
  content: { padding: 20, gap: 12 },
  label: { fontSize: 14, fontWeight: "600", color: "#334155" },
  input: {
    borderWidth: 1,
    borderColor: "#cbd5e1",
    borderRadius: 8,
    padding: 12,
    fontSize: 15,
    backgroundColor: "#fff",
    color: "#1e293b",
  },
  captureRow: { flexDirection: "row", gap: 10, marginTop: 4 },
  captureBtn: {
    flex: 1,
    backgroundColor: "#1e3a5f",
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: "center",
  },
  captureBtnText: { color: "#fff", fontWeight: "600", fontSize: 14 },
  pageCount: { color: "#64748b", fontSize: 13 },
  previewStrip: { gap: 10, paddingVertical: 4 },
  thumbContainer: { alignItems: "center", position: "relative" },
  thumb: { width: 90, height: 120, borderRadius: 6, backgroundColor: "#e2e8f0" },
  thumbLabel: { fontSize: 11, color: "#64748b", marginTop: 4 },
  removeBtn: {
    position: "absolute",
    top: -6,
    right: -6,
    backgroundColor: "#dc2626",
    borderRadius: 10,
    width: 20,
    height: 20,
    alignItems: "center",
    justifyContent: "center",
  },
  removeBtnText: { color: "#fff", fontSize: 11, fontWeight: "700" },
  submitBtn: {
    backgroundColor: "#16a34a",
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: "center",
    marginTop: 12,
  },
  submitBtnDisabled: { backgroundColor: "#94a3b8" },
  submitBtnText: { color: "#fff", fontSize: 16, fontWeight: "600" },
  clearBtn: { alignItems: "center", marginTop: 4 },
  clearBtnText: { color: "#94a3b8", fontSize: 13 },
});
