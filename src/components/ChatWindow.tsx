import React, { useEffect, useRef, useState } from "react";
import { db } from "../firebase";
import { collection, addDoc, query, orderBy, onSnapshot, serverTimestamp } from "firebase/firestore";
import { useAuth } from "../contexts/AuthContext";

interface Message {
  sender: string;
  content: string;
  timestamp: any;
}

export default function ChatWindow({
  conversationId,
  onNewConversation,
}: {
  conversationId: string;
  onNewConversation: (id: string) => void;
}) {
  console.log("ChatWindow rendered");
  const { user } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!user || conversationId === "new") {
      setMessages([]);
      return;
    }
    const q = query(
      collection(db, "conversations", conversationId, "messages"),
      orderBy("timestamp", "asc")
    );
    const unsubscribe = onSnapshot(q, (snapshot) => {
      setMessages(snapshot.docs.map((doc) => doc.data() as Message));
    });
    return () => unsubscribe();
  }, [conversationId, user]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    console.log("sendMessage called", { user, input });
    if (!input.trim() || !user) return;
    let convId = conversationId;
    if (conversationId === "new") {
      console.log("Creating new conversation for user:", user.uid);
      const convRef = await addDoc(collection(db, "conversations"), {
        user_id: user.uid,
        title: input.split(" ").slice(0, 6).join(" "),
        created_at: serverTimestamp(),
      });
      convId = convRef.id;
      onNewConversation(convId);
      console.log("New conversation created:", convId);
    }
    console.log("Adding message to conversation:", convId);
    await addDoc(collection(db, "conversations", convId, "messages"), {
      sender: user.displayName || user.email,
      content: input,
      timestamp: serverTimestamp(),
    });
    console.log("Message added to conversation:", convId);
    setInput("");
    // No need to reload messages, onSnapshot will update automatically
  };

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", height: "100vh" }}>
      <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>
        {messages.map((msg, i) => (
          <div key={i} style={{ margin: "8px 0", textAlign: msg.sender === (user?.displayName || user?.email) ? "right" : "left" }}>
            <span style={{ background: "#eee", padding: 8, borderRadius: 6 }}>{msg.content}</span>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div style={{ padding: 16, borderTop: "1px solid #eee" }}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && sendMessage()}
          style={{ width: "80%", padding: 8 }}
          placeholder="Type your message..."
        />
        <button onClick={() => { console.log("Button clicked"); sendMessage(); }} style={{ marginLeft: 8, padding: 8 }}>Send</button>
      </div>
    </div>
  );
}