"""Export a self-contained, truthful viewer of recorded SUMO trajectories."""

import json
from .data import ROOT

TEMPLATE = r"""<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ATSC · SUMO replay</title>
<style>*{box-sizing:border-box}body{margin:0;background:#0a101b;color:#e4eaf3;font:15px system-ui,sans-serif}main{max-width:1160px;margin:32px auto;padding:0 22px}h1{font-size:30px;margin:8px 0}p{color:#9aaabe;line-height:1.6}.badge{color:#6ee7b7;font:12px monospace;letter-spacing:.1em}.grid{display:grid;grid-template-columns:minmax(300px,760px) 280px;gap:22px}.card{background:#101b2b;border:1px solid #25344b;border-radius:16px;padding:20px}canvas{width:100%;aspect-ratio:1;border-radius:10px;background:#0d1726}button,select{background:#20324b;color:#e4eaf3;border:1px solid #3b526e;border-radius:8px;padding:9px 16px;cursor:pointer}input{width:100%;accent-color:#5de4bf}.controls{display:flex;gap:12px;align-items:center;margin:12px 0}.metric{padding:14px 0;border-bottom:1px solid #243249}.metric b{display:block;font-size:27px;margin-top:4px}.small{font-size:12px;color:#9aaabe}table{width:100%;border-collapse:collapse;margin-top:20px}td,th{padding:9px;text-align:left;border-bottom:1px solid #243249}th{font-size:12px;color:#9aaabe}.legend{display:flex;gap:14px;margin:15px 0;color:#acb8ca;font-size:12px}.dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:5px;background:#62d9b4}@media(max-width:850px){.grid{grid-template-columns:1fr}.card{padding:12px}}a{color:#6ee7b7}</style>
<main><span class="badge">RECORDED SUMO EXPERIMENT</span><h1>Светофор, который учится</h1><p>Траектории из настоящего прогона SUMO. Управление: <strong id="controller"></strong>.<br>Это запись эксперимента; изображение между сохранёнными кадрами интерполируется.</p>
<div class="grid"><section class="card"><canvas id="map" width="900" height="900" aria-label="Движение автомобилей на перекрёстке"></canvas><div class="controls"><button id="play">Пауза</button><select id="speed"><option value="5">5×</option><option value="10">10×</option><option value="20" selected>20×</option><option value="60">60×</option></select><span id="clock"></span></div><input id="timeline" type="range" min="0" step="0.1" value="0" aria-label="Время записи"><div class="legend"><span><i class="dot"></i>Машина движется</span><span><i class="dot" style="background:#f79c64"></i>Медленно / остановка</span></div><p class="small">Показан центральный участок ±220 м; подходы в модели имеют длину 600 м. Пешеходы не моделируются.</p></section>
<aside class="card"><span class="badge">CURRENT FRAME</span><div class="metric">Зелёная фаза<b id="phase"></b></div><div class="metric">Машин в сети<b id="vehicles"></b></div><div class="metric">Очередь на подходах<b id="queue"></b></div><table><tr><th>Подход</th><th>Очередь</th><th>Прогноз*</th></tr><tbody id="approaches"></tbody></table><p class="small">* Среднее число машин за 5 минут в прогнозе на следующие 15 минут. Прогноз обновляется каждые 5 минут.</p><p class="small" id="scenario"></p><p class="small">Ограничения: минимум 10 с зелёного, 3 с жёлтого. В кадрах показано состояние после решения; промежуточный жёлтый интервал в этом разреженном replay не отображается.</p></aside></div><p class="small">Полные метрики, ограничения и инструкции — в RESULTS.html и README_RU.md внутри архива.</p></main>
<script>
const DATA=__DATA__, frames=DATA.frames, names=['Север','Восток','Юг','Запад'];
let t=frames[0].time,playing=true,rate=20,last=performance.now();const canvas=document.getElementById('map'),ctx=canvas.getContext('2d'),W=900,scale=900/440;
const time=document.getElementById('timeline');time.min=frames[0].time;time.max=frames.at(-1).time;document.getElementById('controller').textContent=DATA.controller;document.getElementById('scenario').textContent=`Seed ${DATA.seed} · Номинальная нагрузка ${DATA.saturation} · После 60 минут прогрева.`;
document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'Пауза':'Продолжить';};document.getElementById('speed').onchange=e=>rate=+e.target.value;time.oninput=e=>t=+e.target.value;
function xy(x,y){return [450+x*scale,450-y*scale]}
function draw(now){const dt=Math.min((now-last)/1000,.1);last=now;if(playing)t=Math.min(t+dt*rate,+time.max);time.value=t;
let j=0;while(j<frames.length-1&&frames[j+1].time<=t)j++;let a=frames[j],b=frames[Math.min(j+1,frames.length-1)],mix=Math.max(0,Math.min(1,(t-a.time)/Math.max(1,b.time-a.time)));
ctx.clearRect(0,0,W,W);ctx.fillStyle='#101e2f';ctx.fillRect(0,0,W,W);ctx.fillStyle='#273448';ctx.fillRect(0,450-11*scale,W,22*scale);ctx.fillRect(450-11*scale,0,22*scale,W);
ctx.strokeStyle='#91a5bb44';ctx.lineWidth=1;ctx.setLineDash([9,12]);for(let off of [-6.4,-3.2,3.2,6.4]){ctx.beginPath();ctx.moveTo(0,450+off*scale);ctx.lineTo(450-15*scale,450+off*scale);ctx.moveTo(450+15*scale,450+off*scale);ctx.lineTo(900,450+off*scale);ctx.moveTo(450+off*scale,0);ctx.lineTo(450+off*scale,450-15*scale);ctx.moveTo(450+off*scale,450+15*scale);ctx.lineTo(450+off*scale,900);ctx.stroke()}ctx.setLineDash([]);
ctx.fillStyle='#d8e3ef';ctx.font='18px system-ui';ctx.textAlign='center';ctx.fillText('N',450,28);ctx.fillText('S',450,884);ctx.fillText('W',20,455);ctx.fillText('E',880,455);
let nxt=new Map(b.vehicles.map(v=>[v[0],v]));for(const v of a.vehicles){let vv=nxt.get(v[0]);let x=vv?v[1]+(vv[1]-v[1])*mix:v[1],y=vv?v[2]+(vv[2]-v[2])*mix:v[2];if(Math.abs(x)>222||Math.abs(y)>222)continue;let [px,py]=xy(x,y);ctx.save();ctx.translate(px,py);ctx.rotate((v[3]-90)*Math.PI/180);ctx.fillStyle=v[4]<2?'#f79c64':'#62d9b4';ctx.fillRect(-2.5*scale,-1*scale,5*scale,2*scale);ctx.restore()}
let lamps=[[-12,18],[18,12],[12,-18],[-18,-12]];lamps.forEach((p,i)=>{let [x,y]=xy(...p);ctx.fillStyle=i===a.phase?'#67f0b8':'#ed7283';ctx.beginPath();ctx.arc(x,y,5,0,7);ctx.fill()});
document.getElementById('clock').textContent=`${Math.floor(t/60)}:${String(Math.floor(t%60)).padStart(2,'0')}`;document.getElementById('phase').textContent=names[a.phase];document.getElementById('vehicles').textContent=a.vehicles.length;document.getElementById('queue').textContent=a.queue.reduce((x,y)=>x+y,0);document.getElementById('approaches').innerHTML=names.map((n,i)=>`<tr><td>${n}</td><td>${a.queue[i]}</td><td>${a.forecast[i].toFixed(1)}</td></tr>`).join('');requestAnimationFrame(draw)}requestAnimationFrame(draw);
</script></html>"""


def build():
    data = json.loads((ROOT / "artifacts" / "replay.json").read_text())
    (ROOT / "REPLAY.html").write_text(
        TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"))),
        encoding="utf-8",
    )


if __name__ == "__main__":
    build()
