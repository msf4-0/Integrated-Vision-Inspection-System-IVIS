import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib"

import React, { ReactNode } from "react"
import PropTypes from "prop-types"
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
    headerClassName: "super-app-theme--header",
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
const useStyles = makeStyles({
  root: {
    "& .super-app-theme--header": {
      backgroundColor: "rgba(255, 7, 0, 0.55)",
    },
  },
})
const StyledTable = withStyles({
  cell: {
    background: "linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)",
    color:'white',
    border:'white',
  },

})(DataGrid)

/**
 * This is a React-based component template. The `render()` function is called
 * automatically when your component should be re-rendered.
 */

export default function Table() {
  //   const classes = useStyles()
  return (
    <div style={{ height: 400, width: "100%" }}>
      <StyledTable
        rows={rows}
        columns={columns}
        pageSize={5}
        checkboxSelection
        disableSelectionOnClick
      />
    </div>
  )
}
