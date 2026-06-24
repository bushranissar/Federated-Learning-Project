"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, X, Image as ImageIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface ImageUploaderProps {
  onImageSelect: (file: File) => void;
  onClear: () => void;
  selectedImage: string | null;
}

export default function ImageUploader({ onImageSelect, onClear, selectedImage }: ImageUploaderProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onImageSelect(acceptedFiles[0]);
      }
    },
    [onImageSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".png", ".jpg", ".jpeg", ".webp"] },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  if (selectedImage) {
    return (
      <div className="relative">
        <div className="relative rounded-xl overflow-hidden border-2 border-purple-200 shadow-lg">
          <img
            src={selectedImage}
            alt="Selected tomato leaf"
            className="w-full h-72 object-cover"
          />
          <button
            onClick={onClear}
            className="absolute top-2 right-2 p-1.5 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors shadow-lg"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      {...getRootProps()}
      className={cn(
        "border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200",
        "hover:border-purple-400 hover:bg-purple-50/50",
        isDragActive
          ? "border-purple-500 bg-purple-50"
          : "border-slate-300 bg-white"
      )}
    >
      <input {...getInputProps()} />
      <div className="flex flex-col items-center gap-3">
        <div className={cn(
          "p-3 rounded-full transition-colors",
          isDragActive ? "bg-purple-100" : "bg-slate-100"
        )}>
          {isDragActive ? (
            <ImageIcon className="h-8 w-8 text-purple-600" />
          ) : (
            <Upload className="h-8 w-8 text-slate-400" />
          )}
        </div>
        <div>
          <p className="text-sm font-medium text-slate-700">
            {isDragActive ? "Drop image here" : "Drag & drop tomato leaf image"}
          </p>
          <p className="text-xs text-slate-400 mt-1">
            or click to browse (PNG, JPG up to 10MB)
          </p>
        </div>
      </div>
    </div>
  );
}