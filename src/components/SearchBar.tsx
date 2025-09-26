import React, { useState, useRef } from "react";
import { Paperclip, Send } from "lucide-react";
import "./SearchBar.css";

interface SearchBarProps {
  onSearch: (query: string, files?: File[]) => void;
  placeholder?: string;
  className?: string;
}
function SearchBar({ 
  onSearch, 
  placeholder = "Ask Verion or upload...",
  className = ""
}: SearchBarProps) {
    const [query, setQuery] = useState("");
    const [isFocused, setIsFocused] = useState(false);

    const handleSubmit = (e: React.FormEvent) => {
      e.preventDefault();
      if (!query.trim()) return;
      onSearch(query);
      setQuery("");
    };

    return (
      <div className="chatgpt-searchbar-container">
        <form className="chatgpt-searchbar-form" onSubmit={handleSubmit}>
          <div className={`chatgpt-searchbar ${isFocused ? "focused" : ""}`}>
            <input
              className="chatgpt-searchbar-input"
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Type a message |"
              autoComplete="off"
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
            />
            <button type="submit" className="chatgpt-searchbar-send" aria-label="Send" disabled={!query.trim()}>
              <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 2L11 13" />
                <path d="M21 2L15 21L11 13L2 9L21 2Z" />
              </svg>
            </button>
          </div>
        </form>

  </div>
    );
}

export default SearchBar;
