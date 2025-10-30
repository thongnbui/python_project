import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { FaPaperPlane, FaUser, FaRobot } from "react-icons/fa";
import "./ChatBox.css";

const API_URL = "http://127.0.0.1:8000";

export default function ChatBox() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await axios.post(`${API_URL}/ask`, { prompt: input });
      const botMessage = {
        role: "assistant",
        content: response.data.response,
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "⚠️ Unable to connect to the backend." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="chat-wrapper">
      <div className="chat-header">
        <FaRobot className="header-icon" />
        <h2>AI Assistant</h2>
      </div>

      <div className="chat-container">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`chat-message ${
              msg.role === "user" ? "user-message" : "bot-message"
            }`}
          >
            <div className="avatar">
              {msg.role === "user" ? <FaUser /> : <FaRobot />}
            </div>
            <div className="bubble">{msg.content}</div>
          </div>
        ))}
        {loading && (
          <div className="chat-message bot-message">
            <div className="avatar">
              <FaRobot />
            </div>
            <div className="bubble typing">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      <div className="chat-input-container">
        <textarea
          className="chat-input"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          rows="2"
        />
        <button className="send-btn" onClick={sendMessage}>
          <FaPaperPlane />
        </button>
      </div>
    </div>
  );
}
