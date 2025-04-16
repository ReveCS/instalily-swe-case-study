import React, { useState, useEffect, useRef } from "react";
import "./ChatWindow.css";
import { getAIMessage } from "../api/api"; // Ensure correct path to api.js
import { marked } from "marked";

function ChatWindow() {

  const defaultMessage = [{
    role: "assistant",
    content: "Hi, I'm PartSelect's customer service bot. I can help you with any questions you may have regarding dishwasher and refrigerator parts. How can I help you today?"
  }];

  const [messages, setMessages] = useState(defaultMessage);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false); // Optional: Add loading state

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    // Ensure ref is current before scrolling
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  };

  useEffect(() => {
    // Scroll whenever messages change
    scrollToBottom();
  }, [messages]);

  const handleSend = async (inputValue) => {
    const trimmedInput = inputValue.trim();
    if (trimmedInput !== "") {
      const userMessage = { role: "user", content: trimmedInput };

      // Add user message immediately and clear input
      // Pass the current messages state to the API call as history
      const currentHistory = [...messages, userMessage];
      setMessages(currentHistory);
      setInput("");
      setIsLoading(true); // Start loading

      try {
        // Call API & get the response data object
        // Pass the history *before* the user message was added,
        // as the backend likely expects history + current message separately.
        const responseData = await getAIMessage(trimmedInput, messages);

        let assistantMessage;

        // Check if the response is the success structure from the backend
        if (responseData && responseData.message) {
          assistantMessage = { role: "assistant", content: responseData.message };
        }
        // Check if it's the fallback error structure from api.js
        else if (responseData && responseData.role === 'assistant' && responseData.content) {
           assistantMessage = responseData; // Use the fallback object directly
        }
        // Handle unexpected response format
        else {
           console.error("Received unexpected response format from API:", responseData);
           assistantMessage = { role: "assistant", content: "Sorry, received an unexpected response." };
        }

        // Update messages state with the new assistant message
        // Use the state *after* user message was added (currentHistory) as the base
        setMessages(prevMessages => [...prevMessages, assistantMessage]);

      } catch (error) {
        // This catch is mostly for network errors *before* api.js handles it,
        // as api.js returns a fallback object for backend errors.
        console.error("Error sending message:", error);
        setMessages(prevMessages => [...prevMessages, { role: 'assistant', content: 'Sorry, there was an error connecting to the server.' }]);
      } finally {
        setIsLoading(false); // Stop loading
      }
    }
  };

  return (
    <div className="chat-window">
      <div className="messages-container">
        {messages.map((message, index) => (
          <div key={index} className={`${message.role}-message-container`}>
            {message.content && (
              <div className={`message ${message.role}-message`}>
                {/* Use marked.parse instead of dangerouslySetInnerHTML for potentially safer rendering if needed, but this is common */}
                <div dangerouslySetInnerHTML={{ __html: marked.parse(message.content).replace(/<p>|<\/p>/g, "") }}></div>
              </div>
            )}
          </div>
        ))}
        {isLoading && (
            <div className="assistant-message-container">
                <div className="message assistant-message thinking">...</div>
            </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="input-area">
        <input // Changed back to input
          type="text" // Standard text input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message..."
          onKeyPress={(e) => {
            // Send on Enter key press
            if (e.key === "Enter") {
              e.preventDefault(); // Prevent default form submission if wrapped in form
              handleSend(input);
            }
          }}
          disabled={isLoading} // Disable input while loading
        />
        <button
          className="send-button"
          onClick={() => handleSend(input)}
          disabled={isLoading || input.trim() === ""} // Disable button if loading or input empty
        >
          Send
        </button>
      </div>
    </div>
  );
}

export default ChatWindow;
