/* 
Label Studio (TM)
Copyright (c) 2019-2021 Heartex, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. 

Modifications Copyright (c) 2021 Selangor Human Resource Development Centre. All Rights Reserved.
* 1. Adaptations to Streamlit Component API (https://docs.streamlit.io/en/stable/develop_streamlit_components.html)
*/

import { ReactElement, useEffect, useMemo } from "react";

import {
  Streamlit,
  withStreamlitConnection,
  ComponentProps,
} from "streamlit-component-lib";

import LabelStudio from "label-studio";
import "label-studio/build/static/css/main.css";

interface Args {
  config: string;
  interfaces: string[];
  user: Object;
  task: Object;
}

function LabelStudioEditor({ args }: ComponentProps): ReactElement {
  /* Streamlit Component Arguments */
  const config = args.config;
  const interfaces = args.interfaces;
  const user = args.user;
  const task = args.task;

  /*
  Load arguments into Args interface
   * Args:config, interfaces, user, task
   */
  const LS_args = useMemo(
    (): Args => create_args(config, interfaces, user, task),
    [config, interfaces, user, task]
  );

  /* Function to dynamically calc frame height with 20px offset
   *Args:
   *componentFullHeight[int]: obtained from clientHeight of element
   
   Return:
   *frameHeight[int]: frame height to be set with Streamlit.setFrameHeight()
   */
  function frameHeightCalc(componentFullHeight: number) {
    const offset = 0;
    let frameHeight = componentFullHeight + offset;
    console.log("frameHeight", frameHeight);
    return frameHeight;
  }
  console.log("useMemo", LS_args);

  /* Function to Update Frame Height */
  function updateFrameHeight() {
    /* Get DOM element for LS Editor <div> */
    let canvas = document.getElementsByClassName("App_editor__CIAJZ ls-editor");
    //  "App_editorfs__1aruF ls-editor"
    console.log("Outside", canvas);

    /* Set initial render height in case <div> tag not found for className */
    if (canvas.length === 0) {
      console.log("NULL");

      let frameHeight = 1500;
      // console.log("TEST",canvas[0].clientHeight)
      Streamlit.setFrameHeight(frameHeightCalc(frameHeight));
    } else if (canvas.length !== 0) {
      console.log("NOT NULL");
      /* Obtain clientHeight attribute from HTMLContainer  */
      let client_height = canvas[0].clientHeight;

      Streamlit.setFrameHeight(frameHeightCalc(client_height));
    }
  }

  /* useEffect Hook to render functions when there the component re-renders
   * Does not include explicit dependencies to allow continuos render
   */
  useEffect(() => {
    /* Function to RETURN ANNOTATION RESULTS back to Python Land */
    function return_results(
      /* annotation raw results from Label Studio */
      annotations: any,
      /* Used to indicate status of the annotation process
       * 0(Load),1(Submit),2(Update),3(Delete),4(Skip)
       */
      flag: number = 1,
      /* indicate labelling status `for logging purposes` */
      status: string
    ) {
      let results = {};
      if (annotations !== null) {
        /* Generate serialised JSON of the annotation results */
        results = annotations.serializeAnnotation();
      }
      /* Place 'results' and 'flag' into an array */
      let array = [results, flag];
      /* Return array back to Python Land */
      Streamlit.setComponentValue(array);
      console.log({ status }, "Annotations:", { results }, "Flag", { flag });
    }

    /*
     *Instantiate Label Studio
     */
    // eslint-disable-next-line
    var labelStudio = new LabelStudio("label-studio", {
      config: LS_args.config,
      interfaces: LS_args.interfaces,
      user: LS_args.user,
      task: LS_args.task,

      onLabelStudioLoad: function (LS: any) {
        console.log("Load", LS);
        /* let flag = 0 // Skip Task Flag
        let status = "Load"
        const annotations = null //return nothing */
        // return_results(annotations, flag, status)
        var c = LS.annotationStore.addAnnotation({
          userGenerate: true,
        });
        LS.annotationStore.selectAnnotation(c.id);
      },
      onSubmitAnnotation: function (LS: any, annotations: any) {
        console.log("LS:", { LS });
        let flag = 1; // New Submission Flag
        let status = "New Submission";

        return_results(annotations, flag, status);
      },
      onUpdateAnnotation: function (LS: any, annotations: any) {
        console.log("LS update:", { LS });
        let flag = 2; // Update Submission Flag
        let status = "Update Submission";

        return_results(annotations, flag, status);
      },
      onDeleteAnnotation: function (LS: any, annotations: any) {
        console.log("LS Delete:", { LS });
        let flag = 3; // Update Submission Flag
        let status = "Delete Submission";

        return_results(annotations, flag, status);
      },
      onSkipTask: function (LS: any) {
        console.log("LS Skip:", { LS });
        let flag = 4; // Skip Task Flag
        let status = "Skip Task";
        //return nothing
        let annotations = null;
        return_results(annotations, flag, status);
      },
    });
    updateFrameHeight();
  });

  updateFrameHeight();

  return (
    <div
      style={{ height: "100%", width: "100%", overflow: "visible" }}
      id="label-studio"
    ></div>
  );
}

/* Function to generate LS_args  */
function create_args(
  config: string,
  interfaces: string[],
  user: JSON,
  task: JSON
) {
  const ST_args = {
    config: config,
    interfaces: interfaces,
    user: user,
    task: task,
  };
  console.log("UseMemo", ST_args);
  return ST_args;
}

export default withStreamlitConnection(LabelStudioEditor);
