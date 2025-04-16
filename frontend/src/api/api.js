
export async function getAIMessage(message, history = []) {
  try {
    // Fetch from the Flask backend endpoint
    const response = await fetch("http://localhost:5000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history }),
    });

    if (!response.ok) {
      throw new Error("Failed to fetch from backend");
    }

    const data = await response.json();

    return data;
  } catch (error) {
    console.error("Error calling /api/chat:", error);
    // Return a fallback message so that the UI doesn't break
    return { role: "assistant", content: "Sorry, something went wrong." };
  }
}
