/**
 * Wildcard storage and management utilities using IndexedDB
 */

import { openDB, DBSchema, IDBPDatabase } from 'idb';

export interface WildcardEntry {
  id: string;
  content: string; // Can be a single tag, multiple tags, or a sentence
  createdAt: number;
}

export interface WildcardGroup {
  id: string;
  name: string; // Group name, e.g., "hair_color", "backgrounds"
  entries: WildcardEntry[];
  createdAt: number;
  updatedAt: number;
}

interface WildcardDB extends DBSchema {
  wildcards: {
    key: string;
    value: WildcardGroup;
    indexes: {
      'by-name': string;
      'by-updated': number;
    };
  };
}

const DB_NAME = 'wildcard_db';
const DB_VERSION = 1;
const STORE_NAME = 'wildcards';

let dbPromise: Promise<IDBPDatabase<WildcardDB>> | null = null;

/**
 * Initialize IndexedDB
 */
async function getDB(): Promise<IDBPDatabase<WildcardDB>> {
  if (!dbPromise) {
    dbPromise = openDB<WildcardDB>(DB_NAME, DB_VERSION, {
      upgrade(db) {
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
          store.createIndex('by-name', 'name');
          store.createIndex('by-updated', 'updatedAt');
        }
      },
    });
  }
  return dbPromise;
}

/**
 * Get all wildcard groups
 */
export async function getAllWildcardGroups(): Promise<WildcardGroup[]> {
  try {
    const db = await getDB();
    return await db.getAll(STORE_NAME);
  } catch (error) {
    console.error('Failed to load wildcard groups:', error);
    return [];
  }
}

/**
 * Get a specific wildcard group by ID
 */
export async function getWildcardGroup(id: string): Promise<WildcardGroup | null> {
  try {
    const db = await getDB();
    const group = await db.get(STORE_NAME, id);
    return group || null;
  } catch (error) {
    console.error('Failed to load wildcard group:', error);
    return null;
  }
}

/**
 * Create a new wildcard group
 * Throws error if a group with the same name already exists
 */
export async function createWildcardGroup(name: string): Promise<WildcardGroup> {
  const db = await getDB();

  // Check for duplicate name
  const allGroups = await db.getAll(STORE_NAME);
  const duplicate = allGroups.find(g => g.name === name);
  if (duplicate) {
    throw new Error(`A wildcard group named "${name}" already exists.`);
  }

  const newGroup: WildcardGroup = {
    id: Date.now().toString(),
    name,
    entries: [],
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };

  await db.add(STORE_NAME, newGroup);
  return newGroup;
}

/**
 * Update a wildcard group
 */
export async function updateWildcardGroup(
  id: string,
  updates: Partial<Omit<WildcardGroup, 'id' | 'createdAt'>>
): Promise<WildcardGroup | null> {
  const db = await getDB();
  const existing = await db.get(STORE_NAME, id);

  if (!existing) return null;

  const updated: WildcardGroup = {
    ...existing,
    ...updates,
    id: existing.id,
    createdAt: existing.createdAt,
    updatedAt: Date.now(),
  };

  await db.put(STORE_NAME, updated);
  return updated;
}

/**
 * Delete a wildcard group
 */
export async function deleteWildcardGroup(id: string): Promise<boolean> {
  try {
    const db = await getDB();
    await db.delete(STORE_NAME, id);
    return true;
  } catch (error) {
    console.error('Failed to delete wildcard group:', error);
    return false;
  }
}

/**
 * Add an entry to a wildcard group
 */
export async function addWildcardEntry(
  groupId: string,
  content: string
): Promise<WildcardGroup | null> {
  const group = await getWildcardGroup(groupId);
  if (!group) return null;

  const newEntry: WildcardEntry = {
    id: Date.now().toString(),
    content,
    createdAt: Date.now(),
  };

  group.entries.push(newEntry);
  return await updateWildcardGroup(groupId, { entries: group.entries });
}

/**
 * Update an entry in a wildcard group
 */
export async function updateWildcardEntry(
  groupId: string,
  entryId: string,
  content: string
): Promise<WildcardGroup | null> {
  const group = await getWildcardGroup(groupId);
  if (!group) return null;

  const entryIndex = group.entries.findIndex(e => e.id === entryId);
  if (entryIndex === -1) return null;

  group.entries[entryIndex] = {
    ...group.entries[entryIndex],
    content,
  };

  return await updateWildcardGroup(groupId, { entries: group.entries });
}

/**
 * Delete an entry from a wildcard group
 */
export async function deleteWildcardEntry(
  groupId: string,
  entryId: string
): Promise<WildcardGroup | null> {
  const group = await getWildcardGroup(groupId);
  if (!group) return null;

  group.entries = group.entries.filter(e => e.id !== entryId);
  return await updateWildcardGroup(groupId, { entries: group.entries });
}

/**
 * Get a random entry from a wildcard group
 */
export async function getRandomEntry(groupId: string): Promise<string | null> {
  const group = await getWildcardGroup(groupId);
  if (!group || group.entries.length === 0) return null;

  const randomIndex = Math.floor(Math.random() * group.entries.length);
  return group.entries[randomIndex].content;
}

/**
 * Replace wildcards in prompt with random entries
 * Wildcard format: __groupName__
 * Example: "1girl, __hair_color__ hair, __background__"
 */
export async function replaceWildcardsInPrompt(prompt: string): Promise<string> {
  // Find all wildcards in format __name__ (supports spaces and special characters)
  const wildcardPattern = /__([a-zA-Z0-9_ ]+)__/g;
  let result = prompt;
  const matches = [...prompt.matchAll(wildcardPattern)];

  // Get all groups once
  const allGroups = await getAllWildcardGroups();
  const groupMap = new Map(allGroups.map(g => [g.name, g]));

  for (const match of matches) {
    const groupName = match[1];
    const group = groupMap.get(groupName);

    if (group && group.entries.length > 0) {
      const randomIndex = Math.floor(Math.random() * group.entries.length);
      const replacement = group.entries[randomIndex].content;
      result = result.replace(match[0], replacement);
    }
  }

  return result;
}
