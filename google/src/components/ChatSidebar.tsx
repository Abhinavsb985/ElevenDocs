import React, { useEffect, useState } from "react";
import { db } from "../firebase";
import { collection, query, where, orderBy, getDocs } from "firebase/firestore";
import { useAuth } from "../contexts/AuthContext";

interface Conversation {
  id: string;
  title: string;
  created_at: any;
}

export default function ChatSidebar({
  selectedId,
  onSelect,
}: {
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const { user } = useAuth();
  const [conversations, setConversations] = useState<Conversation[]>([]);

  useEffect(() => {
    if (!user) return;
    const fetchConversations = async () => {
      const q = query(
        collection(db, "conversations"),
        where("user_id", "==", user.uid),
        orderBy("created_at", "desc")
      );
      const snapshot = await getDocs(q);
      setConversations(
        snapshot.docs.map(
          (doc) => ({ id: doc.id, ...doc.data() } as Conversation)
        )
      );
    };
    fetchConversations();
  }, [user]);

  return (
    <aside style={{ width: 250, borderRight: "1px solid #eee", height: "100vh", overflowY: "auto" }}>
      <button onClick={() => onSelect("new")} style={{ width: "100%", margin: "8px 0" }}>+ New Chat</button>
      {conversations.map((conv) => (
        <div
          key={conv.id}
          style={{
            padding: 12,
            background: selectedId === conv.id ? "#f0f0f0" : "transparent",
            cursor: "pointer",
          }}
          onClick={() => onSelect(conv.id)}
        >
          {conv.title}
        </div>
      ))}
    </aside>
  );
}