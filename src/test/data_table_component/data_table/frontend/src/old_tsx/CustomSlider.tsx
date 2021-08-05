import React, { useEffect, ReactNode, ReactElement } from "react"
import {
  Streamlit,
  withStreamlitConnection,
  StreamlitComponentBase,
  ComponentProps,
} from "streamlit-component-lib"
import {
  makeStyles,
  Theme,
  createStyles,
  withStyles,
} from "@material-ui/core/styles"
import Typography from "@material-ui/core/Typography"
import Slider from "@material-ui/core/Slider"

const marks = [
  {
    value: 0,
    label: "0°C",
  },
  {
    value: 20,
    label: "20°C",
  },
  {
    value: 37,
    label: "37°C",
  },
  {
    value: 100,
    label: "100°C",
  },
]

function valuetext(value: number) {
  return `${value}°C`
}

function createMarks(labels: string[]): any[] {
  return labels.map((label, index) => {
    return { value: index, label }
  })
}

function CustomSlider({ args, disabled, theme }: ComponentProps): ReactElement {
  useEffect(() => Streamlit.setFrameHeight())

  console.log(theme)
  interface ComponentTheme {
    primaryColor: string | undefined
    secondaryBackgroundColor: string | undefined
    textColor: string | undefined
    backgroundColor: string | undefined
  }
  // Declare theme from Streamlit Component
  let myTheme: ComponentTheme = {
    primaryColor: theme?.primaryColor,
    secondaryBackgroundColor: theme?.secondaryBackgroundColor,
    textColor: theme?.textColor,
    backgroundColor: theme?.backgroundColor,
  }

  const useStyles = makeStyles({
    root: {
      //   width: 300,
      color: myTheme.textColor,
    },
    margin: {
      height: 50,
      width: 100,
    },
    rail: {
      color: myTheme.primaryColor,
    },
    marked: { color: myTheme.primaryColor, marginTop: -14, marginLeft: -14 },
    thumb: {
      height: 28,
      width: 28,
      backgroundColor: myTheme.secondaryBackgroundColor,
      marginTop: -14,
      marginLeft: -14,
    },
  })

  const PrettoSlider = withStyles({
    root: {
      color: "#52af77",
      height: 8,
    },
    thumb: {
      height: 24,
      width: 24,
      backgroundColor: "#fff",
      border: "2px solid currentColor",
      marginTop: -8,
      marginLeft: -12,
      "&:focus, &:hover, &$active": {
        boxShadow: "inherit",
      },
    },
    active: {},
    valueLabel: {
      left: "calc(-50% + 4px)",
    },
    track: {
      height: 8,
      borderRadius: 4,
    },
    rail: {
      height: 8,
      borderRadius: 4,
    },
  })(Slider)

  const classes = useStyles()
  const vMargin = 7 //vertical margin
  const hMargin = 50 //horizontal margin
  console.log(args.width)

  return (
    <div
      style={{
        width: args.width - hMargin * 2,
        margin: `${vMargin}px ${hMargin}px`,
      }}
      className={classes.root}
    >
      <Typography id="discrete-slider-custom" gutterBottom>
        Custom marks
      </Typography>
      <Slider
        defaultValue={20}
        getAriaValueText={valuetext}
        aria-labelledby="discrete-slider-custom"
        step={10}
        valueLabelDisplay="auto"
        marks={marks}
        classes={{
          root: classes.root,
          rail: classes.rail,
          marked: classes.marked,
          thumb: classes.thumb,
        }}
        disabled={disabled}
      />
      <PrettoSlider
        valueLabelDisplay="auto"
        aria-label="pretto slider"
        defaultValue={20}
        marks={marks}
        getAriaValueText={valuetext}
      />
    </div>
  )
}

export default withStreamlitConnection(CustomSlider)
