"use client";

import { useState, useEffect } from "react";
import Button from "./Button";
import Input from "./Input";
import TextareaWithTagSuggestions from "./TextareaWithTagSuggestions";
import {
  getAllWildcardGroups,
  createWildcardGroup,
  deleteWildcardGroup,
  updateWildcardGroup,
  addWildcardEntry,
  updateWildcardEntry,
  deleteWildcardEntry,
  WildcardGroup,
  WildcardEntry,
} from "@/utils/wildcardStorage";

interface WildcardPanelProps {
  onInsert: (wildcard: string) => void;
}

export default function WildcardPanel({ onInsert }: WildcardPanelProps) {
  const [groups, setGroups] = useState<WildcardGroup[]>([]);
  const [selectedGroupId, setSelectedGroupId] = useState<string | null>(null);
  const [newGroupName, setNewGroupName] = useState("");
  const [newEntryContent, setNewEntryContent] = useState("");
  const [editingEntryId, setEditingEntryId] = useState<string | null>(null);
  const [editingEntryContent, setEditingEntryContent] = useState("");
  const [isCreatingGroup, setIsCreatingGroup] = useState(false);

  // Load all groups
  useEffect(() => {
    loadGroups();
  }, []);

  const loadGroups = async () => {
    const allGroups = await getAllWildcardGroups();
    setGroups(allGroups);
  };

  const selectedGroup = groups.find(g => g.id === selectedGroupId);

  const handleCreateGroup = async () => {
    if (!newGroupName.trim()) {
      alert("Group name cannot be empty");
      return;
    }

    try {
      await createWildcardGroup(newGroupName.trim());
      setNewGroupName("");
      setIsCreatingGroup(false);
      await loadGroups();
    } catch (error: any) {
      alert(error.message || "Failed to create wildcard group");
    }
  };

  const handleDeleteGroup = async (groupId: string) => {
    if (confirm("Are you sure you want to delete this wildcard group?")) {
      await deleteWildcardGroup(groupId);
      if (selectedGroupId === groupId) {
        setSelectedGroupId(null);
      }
      await loadGroups();
    }
  };

  const handleRenameGroup = async (groupId: string, newName: string) => {
    if (!newName.trim()) return;
    await updateWildcardGroup(groupId, { name: newName.trim() });
    await loadGroups();
  };

  const handleAddEntry = async () => {
    if (!selectedGroupId || !newEntryContent.trim()) {
      if (!selectedGroupId) return;
      alert("Entry content cannot be empty");
      return;
    }

    await addWildcardEntry(selectedGroupId, newEntryContent.trim());
    setNewEntryContent("");
    await loadGroups();

    // Blur the textarea to prevent suggestion from appearing
    const textarea = document.activeElement as HTMLTextAreaElement;
    if (textarea && textarea.tagName === "TEXTAREA") {
      textarea.blur();
      // Refocus after a brief delay to allow suggestions to clear
      setTimeout(() => textarea.focus(), 50);
    }
  };

  const handleUpdateEntry = async (entryId: string) => {
    if (!selectedGroupId || !editingEntryContent.trim()) return;

    await updateWildcardEntry(selectedGroupId, entryId, editingEntryContent.trim());
    setEditingEntryId(null);
    setEditingEntryContent("");
    await loadGroups();

    // Blur to prevent suggestion from appearing
    const textarea = document.activeElement as HTMLTextAreaElement;
    if (textarea && textarea.tagName === "TEXTAREA") {
      textarea.blur();
    }
  };

  const handleDeleteEntry = async (entryId: string) => {
    if (!selectedGroupId) return;
    await deleteWildcardEntry(selectedGroupId, entryId);
    await loadGroups();
  };

  const handleInsertWildcard = (groupName: string) => {
    onInsert(`__${groupName}__`);
  };

  const startEditingEntry = (entry: WildcardEntry) => {
    setEditingEntryId(entry.id);
    setEditingEntryContent(entry.content);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-200">Wildcard Manager</h3>
        <Button
          onClick={() => setIsCreatingGroup(!isCreatingGroup)}
          variant="primary"
          size="sm"
        >
          {isCreatingGroup ? "Cancel" : "Create Group"}
        </Button>
      </div>

      {/* Create Group Form */}
      {isCreatingGroup && (
        <div className="bg-gray-800 p-4 rounded-lg space-y-3 border border-gray-700">
          <Input
            label="Group Name"
            value={newGroupName}
            onChange={(e) => setNewGroupName(e.target.value)}
            placeholder="e.g., hair_color, backgrounds"
          />
          <Button onClick={handleCreateGroup} variant="primary" size="sm">
            Create Group
          </Button>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Left: Group List */}
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-gray-300">Wildcard Groups</h4>
          <div className="space-y-2 max-h-[600px] overflow-y-auto">
            {groups.length === 0 ? (
              <div className="text-center text-gray-400 py-8">
                No wildcard groups. Create one to get started!
              </div>
            ) : (
              groups.map(group => (
                <div
                  key={group.id}
                  className={`p-3 rounded-lg border transition-colors cursor-pointer ${
                    selectedGroupId === group.id
                      ? "bg-blue-900 border-blue-600"
                      : "bg-gray-800 border-gray-700 hover:border-gray-600"
                  }`}
                  onClick={() => setSelectedGroupId(group.id)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="font-medium text-gray-200">{group.name}</div>
                      <div className="text-xs text-gray-500">
                        {group.entries.length} entries
                      </div>
                    </div>
                    <div className="flex gap-1">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleInsertWildcard(group.name);
                        }}
                        className="px-2 py-1 text-xs bg-green-600 hover:bg-green-700 text-white rounded"
                        title="Insert wildcard"
                      >
                        Insert
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteGroup(group.id);
                        }}
                        className="px-2 py-1 text-xs bg-red-600 hover:bg-red-700 text-white rounded"
                        title="Delete group"
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Right: Entries List */}
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-gray-300">
            {selectedGroup ? `Entries: ${selectedGroup.name}` : "Select a group"}
          </h4>

          {selectedGroup && (
            <>
              {/* Add Entry Form */}
              <div className="bg-gray-800 p-3 rounded-lg border border-gray-700">
                <TextareaWithTagSuggestions
                  value={newEntryContent}
                  onChange={(e) => setNewEntryContent(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.ctrlKey && e.key === "Enter") {
                      e.preventDefault();
                      handleAddEntry();
                    }
                  }}
                  placeholder="e.g., red, blue hair, standing in garden (Ctrl+Enter to add)"
                  rows={1}
                  enableWeightControl={false}
                />
                <Button onClick={handleAddEntry} variant="primary" size="sm" className="mt-2">
                  Add Entry
                </Button>
              </div>

              {/* Entries */}
              <div className="space-y-2 max-h-[500px] overflow-y-auto">
                {selectedGroup.entries.length === 0 ? (
                  <div className="text-center text-gray-400 py-8">
                    No entries. Add some variations!
                  </div>
                ) : (
                  selectedGroup.entries.map(entry => (
                    <div
                      key={entry.id}
                      className="bg-gray-800 p-3 rounded-lg border border-gray-700 hover:border-gray-600"
                    >
                      {editingEntryId === entry.id ? (
                        <div className="space-y-2">
                          <TextareaWithTagSuggestions
                            value={editingEntryContent}
                            onChange={(e) => setEditingEntryContent(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.ctrlKey && e.key === "Enter") {
                                e.preventDefault();
                                handleUpdateEntry(entry.id);
                              } else if (e.key === "Escape") {
                                setEditingEntryId(null);
                                setEditingEntryContent("");
                              }
                            }}
                            rows={1}
                            enableWeightControl={false}
                          />
                          <div className="flex gap-2">
                            <Button
                              onClick={() => handleUpdateEntry(entry.id)}
                              variant="primary"
                              size="sm"
                            >
                              Save
                            </Button>
                            <Button
                              onClick={() => {
                                setEditingEntryId(null);
                                setEditingEntryContent("");
                              }}
                              variant="secondary"
                              size="sm"
                            >
                              Cancel
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <div className="flex items-start justify-between">
                          <div className="flex-1 text-sm text-gray-300 break-words">
                            {entry.content}
                          </div>
                          <div className="flex gap-1 ml-2">
                            <button
                              onClick={() => startEditingEntry(entry)}
                              className="px-2 py-1 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded"
                            >
                              Edit
                            </button>
                            <button
                              onClick={() => handleDeleteEntry(entry.id)}
                              className="px-2 py-1 text-xs bg-red-600 hover:bg-red-700 text-white rounded"
                            >
                              Delete
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            </>
          )}
        </div>
      </div>

      {/* Usage Info */}
      <div className="mt-4 p-3 bg-gray-900 rounded text-xs text-gray-400">
        <p className="mb-1">
          <strong>Usage:</strong>
        </p>
        <ul className="list-disc list-inside space-y-1">
          <li>Create groups to organize variations (e.g., "hair_color", "backgrounds")</li>
          <li>Add entries: single tags, multiple tags, or sentences</li>
          <li>Click "Insert" to add wildcard placeholder: __group_name__</li>
          <li>When queuing, wildcards are replaced with random entries</li>
          <li>Example: "1girl, __hair_color__ hair" → "1girl, red hair"</li>
        </ul>
      </div>
    </div>
  );
}
