import {
  Streamlit,
  ComponentProps,
  withStreamlitConnection,
} from "streamlit-component-lib";
import { ReactElement, useState } from "react";

interface ComponentTheme {
  primaryColor: string | undefined;
  secondaryBackgroundColor: string | undefined;
  textColor: string | undefined;
  backgroundColor: string | undefined;
  font: string | undefined;
}

function ColorExtract({ args, theme }: ComponentProps): ReactElement {
  const myTheme: ComponentTheme = {
    primaryColor: theme?.primaryColor,
    secondaryBackgroundColor: theme?.secondaryBackgroundColor,
    textColor: theme?.textColor,
    backgroundColor: theme?.backgroundColor,
    font: theme?.font,
  };
  const [currentFlag, setFlag] = useState(0);
  const [currentTheme, setTheme] = useState(myTheme);
  const onThemeChange = (myTheme: ComponentTheme) => {
    setTheme(myTheme);
    setFlag(1);
    Streamlit.setComponentValue(myTheme);
    console.log(myTheme);
  };
  if (
    myTheme.backgroundColor !== currentTheme.backgroundColor ||
    currentFlag === 0
  ) {
    onThemeChange(myTheme);
  }
  console.log("Flag", currentFlag);
  console.log(myTheme);

  return <div>{null}</div>;
}

export default withStreamlitConnection(ColorExtract);
