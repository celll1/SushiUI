/**
 * Template storage and management utilities
 */

export interface Template {
  id: string;
  name: string;
  content: string;
  category: string;
  createdAt: number;
  updatedAt: number;
}

const STORAGE_KEY = 'prompt_templates';

/**
 * Get all templates from localStorage
 */
export function getAllTemplates(): Template[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return [];
    return JSON.parse(stored);
  } catch (error) {
    console.error('Failed to load templates:', error);
    return [];
  }
}

/**
 * Save a new template
 */
export function saveTemplate(name: string, content: string, category: string = 'General'): Template {
  const templates = getAllTemplates();

  const newTemplate: Template = {
    id: Date.now().toString(),
    name,
    content,
    category,
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };

  templates.push(newTemplate);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(templates));

  return newTemplate;
}

/**
 * Update an existing template
 */
export function updateTemplate(id: string, updates: Partial<Template>): Template | null {
  const templates = getAllTemplates();
  const index = templates.findIndex(t => t.id === id);

  if (index === -1) return null;

  templates[index] = {
    ...templates[index],
    ...updates,
    updatedAt: Date.now(),
  };

  localStorage.setItem(STORAGE_KEY, JSON.stringify(templates));
  return templates[index];
}

/**
 * Delete a template
 */
export function deleteTemplate(id: string): boolean {
  const templates = getAllTemplates();
  const filtered = templates.filter(t => t.id !== id);

  if (filtered.length === templates.length) return false;

  localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
  return true;
}

/**
 * Get templates by category
 */
export function getTemplatesByCategory(category: string): Template[] {
  const templates = getAllTemplates();
  return templates.filter(t => t.category === category);
}

/**
 * Get all categories
 */
export function getCategories(): string[] {
  const templates = getAllTemplates();
  const categories = new Set(templates.map(t => t.category));
  return Array.from(categories).sort();
}
