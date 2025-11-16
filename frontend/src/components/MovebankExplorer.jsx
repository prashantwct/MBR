import React, { useState, useRef } from "react";
import Papa from "papaparse";
import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { saveAs } from "file-saver";

export default function MovebankExplorer(){
  const [mbUser, setMbUser] = useState("");
  const [mbPass, setMbPass] = useState("");
  const [rows, setRows] = useState([]);
  const fileRef = useRef();

  async function fetchStudy(studyId){
  try {
    const API_BASE = import.meta.env.VITE_API_BASE || "";
    const res = await fetch(`${API_BASE}/api/download-study`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        study_id: String(studyId),
        username: mbUser || undefined,
        password: mbPass || undefined
      })
    });

    if (!res.ok) {
      const err = await res.json().catch(()=>null);
      throw new Error(err?.detail || res.statusText);
    }

    const text = await res.text();
    const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
    setRows(parsed.data || []);

  } catch (e) {
    alert("Download error: " + e.message);
  }
}


  function handleFile(e){
    const f = e.target.files?.[0];
    if(!f) return;
    Papa.parse(f, { header:true, skipEmptyLines:true, complete: r => setRows(r.data || []) });
  }

  return (<div className="p-4">
    <h1>Movebank Explorer (Frontend)</h1>
    <div style={{marginBottom:10}}>
      <input placeholder="Movebank username" value={mbUser} onChange={e=>setMbUser(e.target.value)} />
      <input placeholder="Movebank password" type="password" value={mbPass} onChange={e=>setMbPass(e.target.value)} />
      <button onClick={()=>{ const id = prompt("Enter study_id:"); if(id) fetchStudy(id); }}>Fetch study</button>
    </div>
    <div style={{marginBottom:10}}>
      <input type="file" ref={fileRef} onChange={handleFile} />
    </div>
    <div>
      <p>Rows: {rows.length}</p>
      <div style={{height:400}}>
        <MapContainer center={[20,0]} zoom={2} style={{height:"100%", width:"100%"}}>
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          {rows.filter(r=>r.location_lat && r.location_long).map((r,i)=>(
            <CircleMarker key={i} center={[parseFloat(r.location_lat), parseFloat(r.location_long)]} radius={3}>
              <Popup>{r.individual_local_identifier || r.individual || "unknown"}</Popup>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>
    </div>
  </div>);
}
