import type { Metadata } from "next";
import "./globals.css";
import Header from "@/components/layout/Header";
import ErrorBoundary from "@/components/common/ErrorBoundary";

export const metadata: Metadata = {
  title: "Federated Learning - Tomato Leaf Disease Detection",
  description:
    "A research dashboard for Federated Learning-based Tomato Leaf Disease Detection using CNN-SVM Ensemble with Flower Framework",
  keywords: [
    "federated learning",
    "tomato disease",
    "CNN",
    "SVM",
    "deep learning",
    "plant disease detection",
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50">
        <ErrorBoundary>
          <Header />
          <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
            {children}
          </main>
        </ErrorBoundary>
      </body>
    </html>
  );
}