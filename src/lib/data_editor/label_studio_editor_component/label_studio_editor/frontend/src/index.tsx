import React from "react"
import ReactDOM from "react-dom"
import LabelStudioEditor from "./LabelStudioEditor"
import "./ls.css"

ReactDOM.render(
  <React.StrictMode>
    <link
      href="https://unpkg.com/label-studio@1.0.1/build/static/css/main.css"
      rel="stylesheet"
    />
    <div id="label-studio">
      <LabelStudioEditor />
    </div>
  </React.StrictMode>,
  document.getElementById("root")
)
