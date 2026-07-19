"use client";

import { Dispatch, ReactNode, SetStateAction, useState } from "react";
import Card from "./Card";
import Button from "./Button";

// Generic single-open tabbed accordion for grouping a generation panel's
// secondary options (ControlNet / Regional Prompt / Seam / Continuity /
// Acceleration / Post-process, etc.) under one Card instead of a stack of
// individual <details> collapsibles.
//
// Ported from OutpaintPanel's original OUTPAINT_OPTIONS_TABS /
// OUTPAINT_OPTIONS_TAB_KEYS / isOutpaintOptionsTabActive /
// outpaintOptionsResetPatch + inline Card render block, generalized so any
// generation panel (Txt2Img/Img2Img/Inpaint/Outpaint) can reuse the same
// chrome: the tab bar (with a green "active" dot when a tab's own
// isActive() predicate is true), the single-open panel body, and a per-tab
// "デフォルトに戻す" button that resets only that tab's keys back to
// defaultParams.
//
// This component owns only the chrome (tab bar / open-state / reset
// button); each tab's body (including its own internal grid layout) is
// supplied by the caller via `render()`.
export interface OptionTab<P> {
  id: string;
  label: string;
  // The param keys owned by this tab. Used only by the tab's own "reset to
  // default" button (resets exactly these keys, not the whole params
  // object) -- NOT used to derive isActive (see below).
  keys: (keyof P)[];
  // "Active" means the group is currently doing something to the
  // generation, not merely "differs from defaultParams". Caller-supplied
  // per-tab, since the definition of "active" varies (e.g. a param whose
  // own default is already non-neutral is still "active" out of the box).
  isActive: (params: P) => boolean;
  // The tab's body content, rendered only while this tab is the single
  // open one.
  render: () => ReactNode;
}

interface TabbedOptionsProps<P> {
  cardTitle: string;
  tabs: OptionTab<P>[];
  params: P;
  setParams: Dispatch<SetStateAction<P>>;
  defaultParams: P;
}

export default function TabbedOptions<P>({
  cardTitle,
  tabs,
  params,
  setParams,
  defaultParams,
}: TabbedOptionsProps<P>) {
  // Which tab is expanded, if any. Single-open accordion: opening one tab
  // closes the others. Starts closed so the options area stays compact
  // until the user opts into a group.
  const [openTabId, setOpenTabId] = useState<string | null>(null);

  const openTab = tabs.find((tab) => tab.id === openTabId) || null;

  return (
    <Card title={cardTitle}>
      <div className="flex flex-wrap gap-1 border-b border-gray-700 -mb-px">
        {tabs.map((tab) => {
          const isTabActive = tab.isActive(params);
          const isTabOpen = openTabId === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => setOpenTabId(isTabOpen ? null : tab.id)}
              className={`px-2.5 py-1.5 text-xs sm:text-sm font-medium transition-colors whitespace-nowrap flex items-center gap-1.5 ${
                isTabOpen ? "border-b-2 border-blue-500 text-white" : "text-gray-400 hover:text-white"
              }`}
              title={isTabActive ? "This group currently has enabled/non-neutral options" : undefined}
            >
              {isTabActive && (
                <span className="w-1.5 h-1.5 rounded-full bg-green-400 flex-shrink-0" aria-hidden="true" />
              )}
              {tab.label}
            </button>
          );
        })}
      </div>

      {openTab && (
        <div className="mt-3 space-y-3">
          <div className="flex justify-end">
            <Button
              onClick={() => {
                const patch: Partial<P> = {};
                for (const key of openTab.keys) {
                  patch[key] = defaultParams[key];
                }
                setParams((prev) => ({ ...prev, ...patch }));
              }}
              variant="secondary"
              size="sm"
            >
              デフォルトに戻す
            </Button>
          </div>

          {openTab.render()}
        </div>
      )}
    </Card>
  );
}
