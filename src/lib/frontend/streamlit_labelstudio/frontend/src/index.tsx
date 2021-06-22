import { Streamlit, RenderData } from "streamlit-component-lib"
import LabelStudio from "label-studio"
import "label-studio/build/static/css/main.css"

const span = document.body.appendChild(document.createElement("span"))
const ls_div = span.appendChild(document.createElement("div"))
ls_div.setAttribute("id", "label-studio")

/**
 * The component's render function. This will be called immediately after
 * the component is initially loaded, and then again every time the
 * component gets new data from Python.
 */
function onRender(event: Event): void {
  // eslint-disable-next-line
  const data = (event as CustomEvent<RenderData>).detail
  // eslint-disable-next-line
  var labelStudio = new LabelStudio("label-studio", {
    config: data.args["config"],
    interfaces: data.args["interfaces"][0],
    user: data.args["user"][0],
    task: data.args["task"],

    onLabelStudioLoad: function (LS) {
      var c = LS.annotationStore.addAnnotation({
        userGenerate: true,
      })
      LS.annotationStore.selectAnnotation(c.id)
    },
    onSubmitAnnotation: function (LS, annotations) {
      console.log("LS:", { LS })
      annotations = JSON.parse(JSON.stringify(annotations)) //JSON.stringify converts JS value to JSON string
      Streamlit.setComponentValue(annotations)
      console.log("Annotations:", { annotations })
      // console.log(annotations.serializeAnnotation())
    },
  })

  // We tell Streamlit to update our frameHeight after each render event, in
  // case it has changed. (This isn't strictly necessary for the example
  // because our height stays fixed, but this is a low-cost function, so
  // there's no harm in doing it redundantly.)
  Streamlit.setFrameHeight()
}

// Attach our `onRender` handler to Streamlit's render event.
Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender)
// Tell Streamlit we're ready to start receiving data. We won't get our
// first RENDER_EVENT until we call this function.
Streamlit.setComponentReady()

// Finally, tell Streamlit to update our initial height. We omit the
// `height` parameter here to have it default to our scrollHeight.
Streamlit.setFrameHeight()
