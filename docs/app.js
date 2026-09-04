const cfg=window.LLM_XRAY_CONFIG||{};let inspection=null,currentTokens=[];const $=id=>document.getElementById(id);const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));const fmt=v=>v==null?'—':Number(v).toFixed(6);const shape=s=>`[${(s||[]).join(' × ')}]`;
const INFO={input:'Layer input hidden state.',input_layernorm:'RMSNorm applied before self-attention.',q_proj:'Query projection.',k_proj:'Key projection.',v_proj:'Value projection.',o_proj:'Attention output projection.',post_attention_layernorm:'RMSNorm applied before the MLP.',gate_proj:'MLP gate projection.',up_proj:'MLP up projection.',down_proj:'MLP down projection.',output:'Layer output passed to the next transformer block.'};
function show(id){$(id).classList.remove('hidden')}function short(n){return n.replace(/^layer_\d+\./,'').replace(/^attention\./,'').replace(/^mlp\./,'')}
function renderModel(d){show('modelCard');$('modelCard').innerHTML=`<h2>Model</h2><div class="model-name">${esc(d.model.name)}</div><div class="model-stats">${d.model.layers} Transformer layers · ${d.model.hidden_size} hidden dimensions · ${d.model.vocab_size.toLocaleString()} vocabulary tokens</div>`}
function renderTokens(d){show('tokenCard');currentTokens=d.tokens||[];let rows=currentTokens.map((t,i)=>`<tr class="token-row" data-i="${i}"><td>${t.position}</td><td>${esc(t.token)}</td><td>${t.token_id}</td><td>${esc(t.token)}</td></tr>`).join('');$('tokenCard').innerHTML=`<h2>1 · Tokenization</h2><div class="tokenizer-info"><div class="tokenizer-stat"><div class="label">Tokenizer</div><div class="value">${esc(d.tokenizer?.class||'Hugging Face tokenizer')}</div></div><div class="tokenizer-stat"><div class="label">Algorithm</div><div class="value">${esc(d.tokenizer?.model_type||'Unknown')}</div></div><div class="tokenizer-stat"><div class="label">Vocabulary</div><div class="value">${(d.tokenizer?.vocab_size||d.model.vocab_size).toLocaleString()}</div></div></div><div class="tokenizer-name">${esc(d.tokenizer?.name||d.model.name)}</div><div class="token-section-title">Tokens</div><div class="token-table-wrapper"><table class="token-table"><thead><tr><th>Position</th><th>Token</th><th>ID</th><th>Decoded text</th></tr></thead><tbody>${rows}</tbody></table></div><div class="embedding-explanation"><div class="embedding-flow" id="embeddingFlow"></div></div><div class="tokenizer-note">Tokenizer markers such as <code>Ġ</code> are tokenizer-specific representations of whitespace.</div>`;document.querySelectorAll('.token-row').forEach(r=>r.onclick=()=>selectToken(+r.dataset.i));selectToken(0)}
function selectToken(i){if(!currentTokens[i])return;document.querySelectorAll('.token-row').forEach(r=>r.classList.toggle('selected',+r.dataset.i===i));let t=currentTokens[i],e=t.embedding||{};$('embeddingFlow').innerHTML=`<div class="embedding-step"><div class="label">Token</div><code>${esc(t.token)}</code></div><div class="embedding-arrow">→</div><div class="embedding-step"><div class="label">Token ID</div><code>${t.token_id}</code></div><div class="embedding-arrow">→</div><div class="embedding-step"><div class="label">Embedding matrix</div><code>${shape(inspection.embedding?.matrix_shape||inspection.embedding?.shape)}</code></div><div class="embedding-arrow">→</div><div class="embedding-step"><div class="label">Selected row</div><code>${e.row??t.token_id}</code></div><div class="embedding-arrow">→</div><div class="embedding-vector"><div class="label">Vector preview</div><code>[${(e.preview||[]).map(fmt).join(', ')} …]</code></div>`}
function renderPipeline(d){show('pipelineCard');let nodes=[['Token IDs',`[${d.tokens.length}]`],['Token embeddings',shape(d.embedding?.sequence_shape||d.embedding?.shape)],['Transformer Layers',`${d.model.layers} layers · hidden size ${d.model.hidden_size}`],['Final RMSNorm (Root Mean Square Normalization)',shape(d.final_hidden_state?.shape)],['LM Head',shape(d.logits?.shape)],['Softmax',`${d.model.vocab_size.toLocaleString()} probabilities`]];$('pipelineCard').innerHTML=`<h2>2 · Transformer pipeline</h2><div class="pipeline">${nodes.map((n,i)=>`<div class="pipeline-box"><div class="pipeline-name">${esc(n[0])}</div><div class="pipeline-shape">${esc(n[1])}</div></div>${i<nodes.length-1?'<div class="arrow">↓</div>':''}`).join('')}</div>`}
function node(e,label,n){return e?`<div class="graph-node captured" data-name="${esc(e.name)}"><div><span class="trace-order">${n}</span>${esc(label)}</div><span class="node-shape">${shape(e.shape)} · ${esc(e.dtype||'')} · ${esc(e.device||'')}</span></div>`:`<div class="graph-node conceptual">${esc(label)} · not captured</div>`}
function graph(entries){let m=Object.fromEntries(entries.map(e=>[short(e.name),e]));let n=0,N=(k,l)=>node(m[k],l,++n);return `<div class="transformer-graph">${N('input','Input')}<div class="arrow">↓ RMSNorm</div>${N('input_layernorm','Input RMSNorm')}<div class="graph-branch"><div>${N('q_proj','Q Projection')}</div><div>${N('k_proj','K Projection')}</div><div>${N('v_proj','V Projection')}</div></div><div class="arrow">↓ Attention → O Projection → Residual</div>${N('o_proj','O Projection')}<div class="arrow">↓</div>${N('post_attention_layernorm','Post-Attention RMSNorm')}<div class="graph-branch"><div>${N('gate_proj','Gate Projection')}</div><div>${N('up_proj','Up Projection')}</div><div class="graph-node conceptual">SiLU × Gate · conceptual</div></div><div class="arrow">↓</div>${N('down_proj','Down Projection')}<div class="arrow">↓ Residual</div>${N('output','Layer Output')}</div>`}
function renderExecution(d){show('executionCard');let traces=d.execution_trace||[];let html='<h2>3 · Transformer execution trace</h2>';for(let i=0;i<d.model.layers;i++){let es=traces.filter(e=>new RegExp(`^layer_${i}\\.`).test(e.name));html+=`<details class="layer" ${i===0?'open':''}><summary class="layer-header"><span class="layer-title">Layer ${i}</span><span class="layer-count">${es.length} captured operations</span></summary><div class="layer-body">${graph(es)}</div></details>`}$('executionCard').innerHTML=html;document.querySelectorAll('.graph-node.captured').forEach(el=>el.onclick=()=>openModal(traces.find(e=>e.name===el.dataset.name)))}
function renderProbabilities(d){show('probabilityCard');$('probabilityCard').innerHTML=`<h2>4 · Next-token probabilities</h2>${(d.next_tokens||[]).map(x=>`<div class="probability-row"><span>${esc(x.token)}</span><div class="prob-bar"><span style="width:${Math.min(100,x.probability*100)}%"></span></div><span>${(x.probability*100).toFixed(2)}%</span></div>`).join('')}`}
let genPaused = false;

function getGenSpeedMultiplier() {
  const slider = document.getElementById('genSpeedSlider');
  return slider ? Number(slider.value) || 1 : 1;
}

function updateGenSpeedLabel() {
  const slider = document.getElementById('genSpeedSlider');
  const label = document.getElementById('genSpeedValue');
  if (slider && label) label.textContent = `${Number(slider.value).toFixed(1)}×`;
}

function genWait(baseMs) {
  return new Promise(resolve => {
    let remaining = baseMs;
    const tickMs = 40;
    function tick() {
      if (remaining <= 0) return resolve();
      setTimeout(() => {
        if (!genPaused) remaining -= tickMs * getGenSpeedMultiplier();
        tick();
      }, tickMs);
    }
    tick();
  });
}

function flowArrow(arrowEl) {
  if (!arrowEl) return;
  arrowEl.classList.remove('flowing');
  void arrowEl.offsetWidth;
  arrowEl.classList.add('flowing');
}

function truncateForNode(text, max = 22) {
  if (!text) return '';
  return text.length <= max ? text : `…${text.slice(-max)}`;
}

async function renderGeneration(data) {
  show('generationCard');

  const stepCount = $('generationStepCount');
  const stepProb = $('generationStepProb');
  const textEl = $('generatedText');
  const contextBox = $('genContextBox');
  const nodeContext = $('genNodeContext');
  const nodeContextValue = $('genNodeContextValue');
  const nodeModel = $('genNodeModel');
  const nodeModelValue = $('genNodeModelValue');
  const nodeToken = $('genNodeToken');
  const nodeTokenValue = $('genNodeTokenValue');
  const arrowIn = $('genArrowIn');
  const arrowOut = $('genArrowOut');
  const loopback = $('genLoopback');

  textEl.innerHTML = '';
  stepProb.textContent = '';
  stepProb.classList.remove('show');
  nodeContext.classList.remove('active');
  nodeModel.classList.remove('active');
  nodeToken.classList.remove('active');
  contextBox.classList.remove('reading');
  loopback.classList.remove('show');
  nodeContextValue.textContent = '—';
  nodeModelValue.textContent = 'idle';
  nodeTokenValue.textContent = '—';

  const steps = Array.isArray(data.generation_steps) ? data.generation_steps : [];
  if (!steps.length) {
    stepCount.textContent = '';
    textEl.innerHTML = '<span class="generation-empty">No generation steps were recorded.</span>';
    return;
  }

  const cursor = document.createElement('span');
  cursor.className = 'gen-cursor';
  textEl.appendChild(cursor);

  const total = steps.length;
  let contextSoFar = '';

  for (let i = 0; i < total; i++) {
    const step = steps[i];
    const probability = Number(step.probability);
    stepCount.textContent = `Token ${i + 1} of ${total}`;

    nodeContextValue.textContent = truncateForNode(contextSoFar) || '(prompt)';
    nodeContext.classList.add('active');
    contextBox.classList.add('reading');
    flowArrow(arrowIn);
    await genWait(260);

    nodeContext.classList.remove('active');
    contextBox.classList.remove('reading');
    nodeModel.classList.add('active');
    nodeModelValue.textContent = 'computing…';
    await genWait(300);

    flowArrow(arrowOut);
    nodeModel.classList.remove('active');
    nodeModelValue.textContent = 'idle';
    nodeToken.classList.add('active');
    nodeTokenValue.textContent = step.token;
    stepProb.textContent = `${(probability * 100).toFixed(1)}% confidence`;
    stepProb.classList.add('show');

    const ghostSpan = document.createElement('span');
    ghostSpan.className = 'gen-token-ghost';
    ghostSpan.textContent = step.token;
    textEl.insertBefore(ghostSpan, cursor);
    await genWait(420);

    ghostSpan.remove();
    const tokenSpan = document.createElement('span');
    tokenSpan.className = 'gen-token';
    tokenSpan.textContent = step.token;
    textEl.insertBefore(tokenSpan, cursor);
    contextSoFar += step.token;

    nodeToken.classList.remove('active');
    loopback.classList.add('show');
    await genWait(480);

    stepProb.classList.remove('show');
    loopback.classList.remove('show');
    await genWait(40);
  }

  cursor.remove();
  nodeContextValue.textContent = truncateForNode(contextSoFar);
  nodeModelValue.textContent = 'idle';
  nodeTokenValue.textContent = '—';
  stepCount.textContent = `Generated ${total} tokens`;
  stepProb.textContent = '';
}

function openModal(e){if(!e)return;let key=short(e.name),stats=e.stats||{},p=e.preview||[];$('modalTitle').textContent=key;$('modalRaw').textContent=e.name;$('modalBody').innerHTML=`<p class="status">${esc(INFO[key]||'Captured intermediate tensor.')}</p><div class="chips"><span class="chip">shape ${shape(e.shape)}</span><span class="chip">dtype ${esc(e.dtype)}</span><span class="chip">device ${esc(e.device)}</span></div><h3>Tensor statistics</h3><div class="stats-grid">${['min','max','mean','std'].map(k=>`<div class="stat"><div class="label">${k}</div><div class="value">${fmt(stats[k])}</div></div>`).join('')}</div><h3>Sample values</h3><div class="bars">${p.map(v=>`<div class="tensor-bar"><span class="v">${fmt(v)}</span><div class="tensor-track"><i style="width:${Math.min(100,Math.abs(v)/(Math.max(...p.map(x=>Math.abs(x)),1))*100)}%"></i></div></div>`).join('')}</div>`;$('traceModal').classList.add('open');$('traceModal').setAttribute('aria-hidden','false')}
function closeModal(){$('traceModal').classList.remove('open');$('traceModal').setAttribute('aria-hidden','true')}
async function inspect(){let b=$('inspectButton');b.disabled=true;$('status').textContent='Replaying recorded inspection…';try{let r=await fetch(cfg.inspectionUrl,{cache:'no-store'});if(!r.ok)throw Error(`HTTP ${r.status}`);inspection=await r.json();$('prompt').value=inspection.prompt||'';renderModel(inspection);renderTokens(inspection);renderPipeline(inspection);renderExecution(inspection);renderProbabilities(inspection);await renderGeneration(inspection);$('status').textContent=`Static snapshot · ${inspection.execution_trace?.length||0} tensor records loaded`}catch(e){$('status').textContent=`Could not load inspection.json: ${e.message}`}finally{b.disabled=false}}
$('subtitle').textContent=cfg.subtitle||'';$('inspectButton').onclick=inspect;$('modalClose').onclick=closeModal;$('traceModal').onclick=e=>{if(e.target===$('traceModal'))closeModal()};document.addEventListener('keydown',e=>{if(e.key==='Escape')closeModal()});$('genSpeedSlider').oninput=updateGenSpeedLabel;$('genLoopDiagram').addEventListener('mouseenter',()=>{genPaused=true;$('genLoopDiagram').classList.add('paused')});$('genLoopDiagram').addEventListener('mouseleave',()=>{genPaused=false;$('genLoopDiagram').classList.remove('paused')});$('replayGenerationButton').onclick=async()=>{if(!inspection)return;const b=$('replayGenerationButton');b.disabled=true;try{await renderGeneration(inspection)}finally{b.disabled=false}};inspect();
