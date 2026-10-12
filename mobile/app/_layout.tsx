import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 2, staleTime: 30_000 },
  },
});

export default function RootLayout() {
  return (
    <QueryClientProvider client={queryClient}>
      <StatusBar style="light" />
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: "#1e3a5f" },
          headerTintColor: "#ffffff",
          headerTitleStyle: { fontWeight: "bold" },
        }}
      >
        <Stack.Screen name="index" options={{ title: "DocumentAgent" }} />
        <Stack.Screen
          name="capture"
          options={{ title: "Capture Answer Sheet" }}
        />
        <Stack.Screen
          name="grading-result"
          options={{ title: "Grading Result" }}
        />
        <Stack.Screen
          name="mark-sheet-review"
          options={{ title: "Mark Sheet Review" }}
        />
        <Stack.Screen
          name="grade-ledger"
          options={{ title: "Grade Ledger" }}
        />
        <Stack.Screen
          name="settings"
          options={{ title: "Settings" }}
        />
      </Stack>
    </QueryClientProvider>
  );
}
