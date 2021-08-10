import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib"

import React, { ReactNode } from "react"
import PropTypes from 'prop-types';
import {
  DataGrid,
  GridRowsProp,
  GridColDef,
  GridValueGetterParams,
} from "@material-ui/data-grid"

import { makeStyles, styled, withStyles } from "@material-ui/styles"
import { classicNameResolver, isPropertyAccessExpression } from "typescript"

interface State {
  numClicks: number
  isFocused: boolean
  page: number
  pageSize: number
}
const rows: GridRowsProp = [
  { id: 1, lastName: "Snow", firstName: "Jon", age: 35 },
  { id: 2, lastName: "Lannister", firstName: "Cersei", age: 42 },
  { id: 3, lastName: "Lannister", firstName: "Jaime", age: 45 },
  { id: 4, lastName: "Stark", firstName: "Arya", age: 16 },
  { id: 5, lastName: "Targaryen", firstName: "Daenerys", age: null },
  { id: 6, lastName: "Melisandre", firstName: null, age: 150 },
  { id: 7, lastName: "Clifford", firstName: "Ferrara", age: 44 },
  { id: 8, lastName: "Frances", firstName: "Rossini", age: 36 },
  { id: 9, lastName: "Roxie", firstName: "Harvey", age: 65 },
  { id: 10, lastName: "Roxie", firstName: "Harvey", age: 65 },
  { id: 11, lastName: "Roxie", firstName: "Harvey", age: 65 },
  { id: 12, lastName: "Roxie", firstName: "Harvey", age: 65 },
  { id: 13, lastName: "Roxie", firstName: "Harvey", age: 65 },
  { id: 14, lastName: "Roxie", firstName: "Harvey", age: 65 },
  { id: 15, lastName: "Roxie", firstName: "Harvey", age: 65 },
  { id: 16, lastName: "Roxie", firstName: "Harvey", age: 65 },
  { id: 17, lastName: "Roxie", firstName: "Harvey", age: 65 },
]

const columns: GridColDef[] = [
  {
    field: "id",
    headerName: "ID",
    headerAlign: "center",
    align: "center",
    flex: 20,
    hideSortIcons: true,
    disableColumnMenu: true,
  },
  {
    field: "firstName",
    headerName: "First name",
    headerAlign: "center",
    align: "center",
    flex: 150,
    hideSortIcons: true,
  },
  {
    field: "lastName",
    headerName: "Last name",
    headerAlign: "center",
    align: "center",
    flex: 150,
    hideSortIcons: true,
  },
  {
    field: "age",
    headerName: "Age",
    headerAlign: "center",
    align: "center",
    type: "number",
    hideSortIcons: true,

    flex: 50,
    resizable: true,
  },
  {
    field: "fullName",
    headerName: "Full name",
    description: "This column has a value getter and is not sortable.",
    headerAlign: "center",
    align: "left",
    hideSortIcons: true,
    sortable: false,
    // width: 160,
    flex: 160,
    resizable: true,
    valueGetter: (params: GridValueGetterParams) =>
      `${params.getValue(params.id, "firstName") || ""} ${
        params.getValue(params.id, "lastName") || ""
      }`,
  },
]


/**
 * This is a React-based component template. The `render()` function is called
 * automatically when your component should be re-rendered.
 */

class DataTable extends StreamlitComponentBase<State> {
  public state = { numClicks: 0, isFocused: false, page: 0, pageSize: 10 }

  public render = (): ReactNode => {
    const data = this.props.args["data"]
   
    // Streamlit sends us a theme object via props that we can use to ensure
    // that our component has visuals that match the active theme in a
    // streamlit app.
    const { theme } = this.props
    console.log({ theme })
    const style: React.CSSProperties = {}

    if (theme) {
      // Use the theme object to style our button border. Alternatively, the
      // theme style is defined in CSS vars.
      // const borderStyling = `1px solid ${
      //   this.state.isFocused ? theme.primaryColor : "gray"
      // }`
      // style.border = borderStyling
      // style.outline = borderStyling
    }


    return (
      <span>
        <div style={{ height: 500, width: "100%" }}>
          <DataGrid 
            autoPageSize
            pagination
            page={this.state.page}
            pageSize={this.state.pageSize}
            onPageChange={this.onPageChange}
            onPageSizeChange={this.onPageSizeChange}
            rows={rows}
            columns={columns}
            rowsPerPageOptions={[5, 10, 20]}
            checkboxSelection
            disableSelectionOnClick
          />
        </div>
      </span>
    )
  }

  private onPageChange = (newPage: number): void => {
    this.setState({ page: newPage })
  }

  private onPageSizeChange = (newPageSize: number) =>
    this.setState({ pageSize: newPageSize })

  /** Click handler for our "Click Me!" button. */
  private onClicked = (): void => {
    // Increment state.numClicks, and pass the new value back to
    // Streamlit via `Streamlit.setComponentValue`.
    this.setState(
      (prevState) => ({ numClicks: prevState.numClicks + 1 }),
      () => Streamlit.setComponentValue(this.state.numClicks)
    )
  }
}

export default withStreamlitConnection(DataTable)
