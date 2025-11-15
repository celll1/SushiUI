"use client";

import { useState, useEffect } from "react";
import Button from "./Button";
import Input from "./Input";
import TextareaWithTagSuggestions from "./TextareaWithTagSuggestions";
import {
  getAllTemplates,
  saveTemplate,
  deleteTemplate,
  updateTemplate,
  getCategories,
  Template,
} from "@/utils/templateStorage";

interface TemplatePanelProps {
  currentPrompt: string;
  onInsert: (content: string) => void;
}

export default function TemplatePanel({ currentPrompt, onInsert }: TemplatePanelProps) {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>("All");
  const [categories, setCategories] = useState<string[]>([]);
  const [isCreating, setIsCreating] = useState(false);
  const [newTemplateName, setNewTemplateName] = useState("");
  const [newTemplateCategory, setNewTemplateCategory] = useState("General");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState("");
  const [searchQuery, setSearchQuery] = useState("");

  // Load templates and categories
  useEffect(() => {
    loadTemplates();
  }, []);

  const loadTemplates = () => {
    const allTemplates = getAllTemplates();
    setTemplates(allTemplates);
    setCategories(["All", ...getCategories()]);
  };

  // Filter templates by category and search query
  const filteredTemplates = templates.filter(template => {
    // Category filter
    const matchesCategory = selectedCategory === "All" || template.category === selectedCategory;

    // Search filter (search in name and content)
    const matchesSearch = searchQuery.trim() === "" ||
      template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      template.content.toLowerCase().includes(searchQuery.toLowerCase());

    return matchesCategory && matchesSearch;
  });

  const handleSaveTemplate = () => {
    if (!currentPrompt.trim()) {
      alert("Current prompt is empty");
      return;
    }

    // Use "Untitled(date-time)" if name is empty
    const templateName = newTemplateName.trim() ||
      `Untitled ${new Date().toLocaleString('ja-JP', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
      }).replace(/\//g, '-')}`;

    saveTemplate(templateName, currentPrompt, newTemplateCategory);
    setNewTemplateName("");
    setNewTemplateCategory("General");
    setIsCreating(false);
    loadTemplates();
  };

  const handleDeleteTemplate = (id: string) => {
    if (confirm("Are you sure you want to delete this template?")) {
      deleteTemplate(id);
      loadTemplates();
    }
  };

  const handleRenameTemplate = (id: string) => {
    if (!editingName.trim()) return;
    updateTemplate(id, { name: editingName });
    setEditingId(null);
    setEditingName("");
    loadTemplates();
  };

  const startEditing = (template: Template) => {
    setEditingId(template.id);
    setEditingName(template.name);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-200">Template Manager</h3>
        <Button
          onClick={() => setIsCreating(!isCreating)}
          variant="primary"
          size="sm"
        >
          {isCreating ? "Cancel" : "Save Current Prompt"}
        </Button>
      </div>

      {/* Save Template Form */}
      {isCreating && (
        <div className="bg-gray-800 p-4 rounded-lg space-y-3 border border-gray-700">
          <Input
            label="Template Name"
            value={newTemplateName}
            onChange={(e) => setNewTemplateName(e.target.value)}
            placeholder="e.g., Character Portrait"
          />
          <Input
            label="Category"
            value={newTemplateCategory}
            onChange={(e) => setNewTemplateCategory(e.target.value)}
            placeholder="e.g., General, Character, Background"
          />
          <div className="text-xs text-gray-400 bg-gray-900 p-2 rounded max-h-20 overflow-auto">
            {currentPrompt || "(Current prompt is empty)"}
          </div>
          <Button onClick={handleSaveTemplate} variant="primary" size="sm">
            Save Template
          </Button>
        </div>
      )}

      {/* Category Filter and Search */}
      <div className="flex items-start gap-2">
        <div className="flex gap-2 flex-wrap flex-1">
          {categories.map(category => (
            <button
              key={category}
              onClick={() => setSelectedCategory(category)}
              className={`px-3 py-1 rounded text-sm ${
                selectedCategory === category
                  ? "bg-blue-600 text-white"
                  : "bg-gray-700 text-gray-300 hover:bg-gray-600"
              }`}
            >
              {category}
            </button>
          ))}
        </div>
        <div className="w-64">
          <TextareaWithTagSuggestions
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search..."
            rows={1}
            enableWeightControl={false}
          />
        </div>
      </div>

      {/* Template List */}
      <div className="space-y-2 max-h-[600px] overflow-y-auto">
        {filteredTemplates.length === 0 ? (
          <div className="text-center text-gray-400 py-8">
            No templates found. Save your current prompt to create one!
          </div>
        ) : (
          filteredTemplates.map(template => (
            <div
              key={template.id}
              className="bg-gray-800 p-3 rounded-lg border border-gray-700 hover:border-gray-600 transition-colors"
            >
              <div className="flex items-start justify-between mb-2">
                {editingId === template.id ? (
                  <input
                    type="text"
                    value={editingName}
                    onChange={(e) => setEditingName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleRenameTemplate(template.id);
                      if (e.key === "Escape") {
                        setEditingId(null);
                        setEditingName("");
                      }
                    }}
                    onBlur={() => handleRenameTemplate(template.id)}
                    className="flex-1 bg-gray-900 text-gray-100 px-2 py-1 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
                    autoFocus
                  />
                ) : (
                  <div className="flex-1">
                    <h4
                      className="font-medium text-gray-200 cursor-pointer hover:text-blue-400"
                      onClick={() => startEditing(template)}
                    >
                      {template.name}
                    </h4>
                    <span className="text-xs text-gray-500">{template.category}</span>
                  </div>
                )}
                <div className="flex gap-1 ml-2">
                  <button
                    onClick={() => onInsert(template.content)}
                    className="px-2 py-1 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded"
                    title="Insert at cursor position"
                  >
                    Insert
                  </button>
                  <button
                    onClick={() => handleDeleteTemplate(template.id)}
                    className="px-2 py-1 text-xs bg-red-600 hover:bg-red-700 text-white rounded"
                    title="Delete template"
                  >
                    Delete
                  </button>
                </div>
              </div>
              <div className="text-xs text-gray-400 bg-gray-900 p-2 rounded max-h-16 overflow-auto">
                {template.content}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
