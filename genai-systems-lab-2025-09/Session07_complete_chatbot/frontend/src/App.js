import React from "react";
import ChatBox from "./components/ChatBox";

function App() {
  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center">
      <div className="bg-white shadow-xl rounded-2xl p-6 w-full max-w-3xl">
        <h1 className="text-2xl font-bold mb-4 text-center text-indigo-700">
          💬 Fine-tuned GPT-4 Chat
        </h1>
        <ChatBox />
      </div>
    </div>
  );
}

export default App;
