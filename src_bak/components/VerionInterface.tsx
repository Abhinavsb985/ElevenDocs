import { useState } from "react";
import { VerionSidebar } from "./VerionSidebar";
import { VerionHeader } from "./VerionHeader";
import { VerionChat } from "./VerionChat";

export const VerionInterface = () => {
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  return (
    <div className="flex h-screen w-full bg-gradient-bg overflow-hidden">
      {/* Sidebar */}
      <VerionSidebar 
        collapsed={isSidebarCollapsed}
        onToggle={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
      />
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        <VerionHeader 
          onToggleSidebar={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
          isProcessing={isProcessing}
        />
        <VerionChat onProcessingChange={setIsProcessing} />
      </div>
    </div>
  );
};