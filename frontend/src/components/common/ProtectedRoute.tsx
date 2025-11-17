"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/contexts/AuthContext";

export default function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isAuthEnabled, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    // If auth is enabled and user is not authenticated, redirect to login
    if (!isLoading && isAuthEnabled && !isAuthenticated) {
      router.push("/login");
    }
  }, [isAuthenticated, isAuthEnabled, isLoading, router]);

  // Show loading state
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-white">Loading...</div>
      </div>
    );
  }

  // If auth is enabled but not authenticated, show nothing (redirect will happen)
  if (isAuthEnabled && !isAuthenticated) {
    return null;
  }

  // User is authenticated or auth is disabled
  return <>{children}</>;
}
