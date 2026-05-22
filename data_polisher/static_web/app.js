const statusText = {
  waiting: "等待中",
  running: "处理中",
  done: "完成",
  failed: "失败",
};

const modeText = {
  detail: "详细数据",
  eye: "小眼睛",
};

const state = {
  mode: "detail",
  tasks: [],
  selectedId: null,
  sequence: 0,
};

const els = {
  addFiles: document.querySelector("#add-files"),
  startAll: document.querySelector("#start-all"),
  clearTasks: document.querySelector("#clear-tasks"),
  removeSelected: document.querySelector("#remove-selected"),
  saveSelected: document.querySelector("#save-selected"),
  batchSave: document.querySelector("#batch-save"),
  taskBody: document.querySelector("#task-body"),
  summary: document.querySelector("#summary"),
  selectedName: document.querySelector("#selected-name"),
  originalStage: document.querySelector("#original-stage"),
  resultStage: document.querySelector("#result-stage"),
  toast: document.querySelector("#toast"),
  segmentButtons: [...document.querySelectorAll(".segment-button")],
  panels: [...document.querySelectorAll(".mode-panel")],
  exposureLo: document.querySelector("#exposure-lo"),
  exposureHi: document.querySelector("#exposure-hi"),
  viewsLo: document.querySelector("#views-lo"),
  viewsHi: document.querySelector("#views-hi"),
  eyeTitle: document.querySelector("#eye-title"),
  eyeViewsLo: document.querySelector("#eye-views-lo"),
  eyeViewsHi: document.querySelector("#eye-views-hi"),
};

let toastTimer = 0;

function makeId() {
  state.sequence += 1;
  const suffix = String(state.sequence).padStart(4, "0");
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
    return `${globalThis.crypto.randomUUID()}-${suffix}`;
  }
  return `${Date.now()}-${suffix}`;
}

function bridge() {
  if (!window.pywebview || !window.pywebview.api) {
    throw new Error("桌面桥接尚未就绪");
  }
  return window.pywebview.api;
}

function parseRange(lo, hi, label, floor) {
  const left = String(lo).trim();
  const right = String(hi).trim();
  if (!left || !right) {
    throw new Error(`${label}：请填写最小值与最大值`);
  }
  const a = Number.parseInt(left, 10);
  const b = Number.parseInt(right, 10);
  if (!Number.isFinite(a) || !Number.isFinite(b)) {
    throw new Error(`${label} 须为整数`);
  }
  const min = Math.min(a, b);
  const max = Math.max(a, b);
  if (min < floor || max < floor) {
    throw new Error(`${label} 不能小于 ${floor}`);
  }
  return [min, max];
}

function configForMode(mode) {
  if (mode === "eye") {
    const title = els.eyeTitle.value.trim();
    if (!title) {
      throw new Error("标题关键词不能为空");
    }
    const [eyeViewsLo, eyeViewsHi] = parseRange(
      els.eyeViewsLo.value,
      els.eyeViewsHi.value,
      "浏览量范围",
      0,
    );
    return { title, eyeViewsLo, eyeViewsHi };
  }
  const [exposureLo, exposureHi] = parseRange(
    els.exposureLo.value,
    els.exposureHi.value,
    "新曝光数",
    1,
  );
  const [viewsLo, viewsHi] = parseRange(els.viewsLo.value, els.viewsHi.value, "新观看数", 0);
  return { exposureLo, exposureHi, viewsLo, viewsHi };
}

function showToast(message) {
  window.clearTimeout(toastTimer);
  els.toast.textContent = message;
  els.toast.classList.add("is-visible");
  toastTimer = window.setTimeout(() => {
    els.toast.classList.remove("is-visible");
  }, 3200);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function selectedTask() {
  return state.tasks.find((task) => task.id === state.selectedId) || null;
}

function setMode(mode) {
  state.mode = mode;
  for (const button of els.segmentButtons) {
    button.classList.toggle("is-active", button.dataset.mode === mode);
  }
  for (const panel of els.panels) {
    panel.classList.toggle("is-hidden", panel.dataset.panel !== mode);
  }
}

function renderTasks() {
  if (state.tasks.length === 0) {
    els.taskBody.innerHTML = '<tr class="empty-row"><td colspan="4">暂无任务</td></tr>';
    renderSummary();
    renderPreview();
    renderButtons();
    return;
  }

  els.taskBody.innerHTML = state.tasks
    .map((task) => {
      const selected = task.id === state.selectedId ? " is-selected" : "";
      return `
        <tr class="${selected}" data-id="${escapeHtml(task.id)}">
          <td title="${escapeHtml(task.path)}">${escapeHtml(task.name)}</td>
          <td><span class="mode-chip">${modeText[task.mode] || task.mode}</span></td>
          <td><span class="status ${task.status}">${statusText[task.status] || task.status}</span></td>
          <td title="${escapeHtml(task.progress || "")}">${escapeHtml(task.progress || "")}</td>
        </tr>
      `;
    })
    .join("");
  renderSummary();
  renderPreview();
  renderButtons();
}

function renderSummary() {
  const total = state.tasks.length;
  if (!total) {
    els.summary.textContent = "任务列表为空";
    return;
  }
  const done = state.tasks.filter((task) => task.status === "done").length;
  const running = state.tasks.filter((task) => task.status === "running").length;
  const failed = state.tasks.filter((task) => task.status === "failed").length;
  const waiting = state.tasks.filter((task) => task.status === "waiting").length;
  if (running) {
    els.summary.textContent = `处理中… ${done}/${total} 已完成`;
    return;
  }
  if (done === total) {
    els.summary.textContent = `全部完成 · ${done} 张\n可批量保存结果`;
    return;
  }
  const parts = [];
  if (waiting) parts.push(`等待 ${waiting}`);
  if (done) parts.push(`完成 ${done}`);
  if (failed) parts.push(`失败 ${failed}`);
  els.summary.textContent = parts.join(" · ");
}

function renderButtons() {
  const task = selectedTask();
  const hasRunning = state.tasks.some((item) => item.status === "running");
  const hasPending = state.tasks.some((item) => item.status === "waiting");
  const hasDone = state.tasks.some((item) => item.status === "done");

  els.startAll.disabled = hasRunning || !hasPending;
  els.clearTasks.disabled = state.tasks.length === 0;
  els.removeSelected.disabled = !task || task.status === "running";
  els.saveSelected.disabled = !task || task.status !== "done";
  els.batchSave.disabled = !hasDone;
}

function stageImage(stage, src, emptyText) {
  if (!src) {
    stage.innerHTML = `<span>${escapeHtml(emptyText)}</span>`;
    return;
  }
  stage.innerHTML = `<img src="${src}" alt="" />`;
}

function renderPreview() {
  const task = selectedTask();
  if (!task) {
    els.selectedName.textContent = "未选择";
    stageImage(els.originalStage, "", "暂无");
    stageImage(els.resultStage, "", "暂无");
    return;
  }
  els.selectedName.textContent = task.name;
  stageImage(els.originalStage, task.originalPreview, "暂无");
  const resultText = task.status === "running" ? task.progress || "处理中" : "尚未生成";
  stageImage(els.resultStage, task.resultPreview, resultText);
}

function taskSpecsForPending() {
  return state.tasks
    .filter((task) => task.status === "waiting")
    .map((task) => ({
      id: task.id,
      path: task.path,
      name: task.name,
      mode: task.mode,
      config: configForMode(task.mode),
    }));
}

function updateTask(id, patch) {
  const task = state.tasks.find((item) => item.id === id);
  if (!task) {
    return;
  }
  Object.assign(task, patch);
  renderTasks();
}

async function handleAddFiles() {
  try {
    configForMode(state.mode);
    const result = await bridge().select_images();
    if (!result.ok) {
      showToast(result.error || "选择图片失败");
      return;
    }
    const files = result.files || [];
    if (!files.length) {
      return;
    }
    for (const file of files) {
      const task = {
        id: makeId(),
        path: file.path,
        name: file.name,
        mode: state.mode,
        status: "waiting",
        progress: "",
        originalPreview: file.preview || "",
        resultPreview: "",
      };
      state.tasks.push(task);
      state.selectedId = task.id;
    }
    renderTasks();
  } catch (error) {
    showToast(error.message || String(error));
  }
}

async function handleStartAll() {
  try {
    const specs = taskSpecsForPending();
    if (!specs.length) {
      showToast("没有等待中的任务。");
      return;
    }
    const result = await bridge().start_tasks(specs);
    if (!result.ok) {
      showToast(result.error || "启动失败");
    }
  } catch (error) {
    showToast(error.message || String(error));
  }
}

async function handleClearTasks() {
  const hasRunning = state.tasks.some((task) => task.status === "running");
  if (hasRunning && !window.confirm("正在处理，确定要清空任务列表并停止吗？")) {
    return;
  }
  try {
    await bridge().clear_tasks();
  } catch (error) {
    showToast(error.message || String(error));
  }
  state.tasks = [];
  state.selectedId = null;
  renderTasks();
}

async function handleRemoveSelected() {
  const task = selectedTask();
  if (!task || task.status === "running") {
    return;
  }
  try {
    await bridge().remove_task(task.id);
  } catch (error) {
    showToast(error.message || String(error));
  }
  state.tasks = state.tasks.filter((item) => item.id !== task.id);
  state.selectedId = state.tasks[0]?.id || null;
  renderTasks();
}

async function handleSaveSelected() {
  const task = selectedTask();
  if (!task || task.status !== "done") {
    return;
  }
  try {
    const result = await bridge().save_task(task.id);
    if (!result.ok) {
      showToast(result.error || "保存失败");
    } else if (!result.cancelled) {
      showToast("已保存");
    }
  } catch (error) {
    showToast(error.message || String(error));
  }
}

async function handleBatchSave() {
  const ids = state.tasks.filter((task) => task.status === "done").map((task) => task.id);
  if (!ids.length) {
    return;
  }
  try {
    const result = await bridge().batch_save(ids);
    if (!result.ok) {
      showToast(result.error || "批量保存失败");
    } else if (!result.cancelled) {
      showToast(`已保存 ${result.saved || 0} 张`);
    }
  } catch (error) {
    showToast(error.message || String(error));
  }
}

window.DataPolisher = {
  onNativeEvent(event) {
    if (!event || !event.type) {
      return;
    }
    if (event.type === "running") {
      updateTask(event.id, { status: "running", progress: event.progress || "启动…" });
      return;
    }
    if (event.type === "progress") {
      updateTask(event.id, { progress: event.progress || "" });
      return;
    }
    if (event.type === "done") {
      const task = state.tasks.find((item) => item.id === event.id);
      updateTask(event.id, {
        status: "done",
        progress: event.progress || "完成",
        originalPreview: event.originalPreview || task?.originalPreview || "",
        resultPreview: event.resultPreview || "",
      });
      return;
    }
    if (event.type === "failed") {
      updateTask(event.id, {
        status: "failed",
        progress: event.progress || event.error || "失败",
        error: event.error || "",
      });
      return;
    }
    if (event.type === "saved") {
      updateTask(event.id, { progress: event.progress || "已保存" });
      return;
    }
    if (event.type === "finished") {
      renderTasks();
    }
  },
};

els.segmentButtons.forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

els.taskBody.addEventListener("click", (event) => {
  const row = event.target.closest("tr[data-id]");
  if (!row) {
    return;
  }
  state.selectedId = row.dataset.id;
  renderTasks();
});

els.addFiles.addEventListener("click", handleAddFiles);
els.startAll.addEventListener("click", handleStartAll);
els.clearTasks.addEventListener("click", handleClearTasks);
els.removeSelected.addEventListener("click", handleRemoveSelected);
els.saveSelected.addEventListener("click", handleSaveSelected);
els.batchSave.addEventListener("click", handleBatchSave);

setMode("detail");
renderTasks();
