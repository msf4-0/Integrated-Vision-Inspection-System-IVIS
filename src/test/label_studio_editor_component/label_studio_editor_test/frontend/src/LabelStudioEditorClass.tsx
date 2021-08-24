import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
  ComponentProps,
} from "streamlit-component-lib"
import React, { ReactElement, ReactNode } from "react"
import LabelStudio from "label-studio"
import "label-studio/build/static/css/main.css"

interface State {
  flag: number
  status: string
  frameHeight: number
}
interface Args {
  config: string
  interfaces: string[]
  user: Object
  task: Object
}
class LabelStudioEditorClass extends StreamlitComponentBase<State> {
  public state = { flag: 1, status: "Load", frameHeight: 1500 }

  public render = (): ReactNode => {
    // Get component props
    const LS_args: Args = {
      config: this.props.args.config,
      interfaces: this.props.args.interfaces[0],
      user: this.props.args.user[0],
      task: this.props.args.task[0],
    }

    //method to calc frameHeight
    function frameHeightCalc(componentFullHeight: number) {
      const offset = 20
      let frameHeight = componentFullHeight + offset
      console.log("frameHeight", frameHeight)
      return frameHeight
    }

    // Streamlit.setFrameHeight(this.state.frameHeight)

    //return Annotation results back to Streamlit Python land
    function return_results(
      annotations: any,
      flag: number = 1,
      status: string
    ) {
      let results = annotations.serializeAnnotation() //JSON.stringify converts JS value to JSON string
      let array = [results, flag]
      Streamlit.setComponentValue(array)
      console.log({ status }, "Annotations:", { results })
    };

    // Instantiate Label Studio
    

    var labelStudio = new LabelStudio("label-studio", {
      config: LS_args.config,
      interfaces: LS_args.interfaces,
      user: LS_args.user,
      task: LS_args.task,

      onLabelStudioLoad: function (LS: any) {
        console.log("Load", LS)
        let flag = 0 // Skip Task Flag
        let status = "Load"
        const annotations = {} as any //return nothing
        // return_results(annotations, flag, status)
        var c = LS.annotationStore.addAnnotation({
          userGenerate: true,
        })
        LS.annotationStore.selectAnnotation(c.id)
      },
      onSubmitAnnotation: function (LS: any, annotations: any) {
        console.log("LS:", { LS })
        let flag = 1 // New Submission Flag
        let status = "New Submission"

        return_results(annotations, flag, status)
      },
      onUpdateAnnotation: function (LS: any, annotations: any) {
        console.log("LS update:", { LS })
        let flag = 2 // Update Submission Flag
        let status = "Update Submission"

        return_results(annotations, flag, status)
      },
      onDeleteAnnotation: function (LS: any, annotations: any) {
        console.log("LS Delete:", { LS })
        let flag = 3 // Update Submission Flag
        let status = "Delete Submission"
        return_results(annotations, flag, status)
      },
      onSkipTask: function (LS: any) {
        console.log("LS Delete:", { LS })
        let flag = 4 // Skip Task Flag
        let status = "Skip Task"
        //return nothing
        let annotations = {} as any
        return_results(annotations, flag, status)
      },
    })

    console.log("LABELSTUDIO", labelStudio)

    let canvas = document.getElementsByClassName(
      "App_editor__CIAJZ ls-editor"
    )
    console.log(canvas)
    // this.state.frameHeight = canvas[0].clientHeight
    console.log("inside height", this.state.frameHeight)
    this.state.frameHeight = frameHeightCalc(this.state.frameHeight)

    Streamlit.setFrameHeight(1500)

    return (
      <div
        style={{ height: "100%", width: "100%", overflow: "visible" }}
        id="label-studio"
      ></div>
    )
    }
}

export default withStreamlitConnection(LabelStudioEditorClass)
