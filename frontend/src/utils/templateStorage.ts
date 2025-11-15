/**
 * Template storage and management utilities using IndexedDB
 */

import { openDB, DBSchema, IDBPDatabase } from 'idb';

export interface Template {
  id: string;
  name: string;
  content: string;
  category: string;
  createdAt: number;
  updatedAt: number;
}

interface TemplateDB extends DBSchema {
  templates: {
    key: string;
    value: Template;
    indexes: {
      'by-category': string;
      'by-name': string;
      'by-updated': number;
    };
  };
}

const DB_NAME = 'prompt_editor_db';
const DB_VERSION = 1;
const STORE_NAME = 'templates';
const LEGACY_STORAGE_KEY = 'prompt_templates';

let dbPromise: Promise<IDBPDatabase<TemplateDB>> | null = null;

/**
 * Initialize IndexedDB
 */
async function getDB(): Promise<IDBPDatabase<TemplateDB>> {
  if (!dbPromise) {
    dbPromise = openDB<TemplateDB>(DB_NAME, DB_VERSION, {
      upgrade(db) {
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
          store.createIndex('by-category', 'category');
          store.createIndex('by-name', 'name');
          store.createIndex('by-updated', 'updatedAt');
        }
      },
    });

    // Migrate from localStorage if needed
    await migrateFromLocalStorage();
  }
  return dbPromise;
}

/**
 * Migrate data from localStorage to IndexedDB (one-time operation)
 */
async function migrateFromLocalStorage(): Promise<void> {
  try {
    const legacyData = localStorage.getItem(LEGACY_STORAGE_KEY);
    if (!legacyData) return;

    const templates: Template[] = JSON.parse(legacyData);
    if (!templates || templates.length === 0) return;

    const db = await dbPromise;
    if (!db) return;

    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);

    // Check if migration already done (store is not empty)
    const count = await store.count();
    if (count > 0) {
      // Already migrated, remove legacy data
      localStorage.removeItem(LEGACY_STORAGE_KEY);
      return;
    }

    // Migrate all templates
    for (const template of templates) {
      await store.add(template);
    }

    await tx.done;

    // Remove legacy data after successful migration
    localStorage.removeItem(LEGACY_STORAGE_KEY);
    console.log(`Migrated ${templates.length} templates from localStorage to IndexedDB`);
  } catch (error) {
    console.error('Failed to migrate from localStorage:', error);
  }
}

/**
 * Get all templates from IndexedDB
 */
export async function getAllTemplates(): Promise<Template[]> {
  try {
    const db = await getDB();
    return await db.getAll(STORE_NAME);
  } catch (error) {
    console.error('Failed to load templates:', error);
    return [];
  }
}

/**
 * Save a new template
 */
export async function saveTemplate(
  name: string,
  content: string,
  category: string = 'General'
): Promise<Template> {
  const db = await getDB();

  const newTemplate: Template = {
    id: Date.now().toString(),
    name,
    content,
    category,
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };

  await db.add(STORE_NAME, newTemplate);
  return newTemplate;
}

/**
 * Update an existing template
 */
export async function updateTemplate(
  id: string,
  updates: Partial<Template>
): Promise<Template | null> {
  const db = await getDB();
  const existing = await db.get(STORE_NAME, id);

  if (!existing) return null;

  const updated: Template = {
    ...existing,
    ...updates,
    id: existing.id, // Ensure ID doesn't change
    createdAt: existing.createdAt, // Preserve creation time
    updatedAt: Date.now(),
  };

  await db.put(STORE_NAME, updated);
  return updated;
}

/**
 * Delete a template
 */
export async function deleteTemplate(id: string): Promise<boolean> {
  try {
    const db = await getDB();
    await db.delete(STORE_NAME, id);
    return true;
  } catch (error) {
    console.error('Failed to delete template:', error);
    return false;
  }
}

/**
 * Get templates by category
 */
export async function getTemplatesByCategory(category: string): Promise<Template[]> {
  const db = await getDB();
  return await db.getAllFromIndex(STORE_NAME, 'by-category', category);
}

/**
 * Get all categories
 */
export async function getCategories(): Promise<string[]> {
  const templates = await getAllTemplates();
  const categories = new Set(templates.map(t => t.category));
  return Array.from(categories).sort();
}

/**
 * Search templates by name or content
 */
export async function searchTemplates(query: string): Promise<Template[]> {
  if (!query.trim()) {
    return await getAllTemplates();
  }

  const templates = await getAllTemplates();
  const lowerQuery = query.toLowerCase();

  return templates.filter(
    t =>
      t.name.toLowerCase().includes(lowerQuery) ||
      t.content.toLowerCase().includes(lowerQuery)
  );
}
