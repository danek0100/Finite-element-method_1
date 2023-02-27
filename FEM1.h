#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>

using namespace dealii;

template <int dim>
class FEM
{
  public:
    FEM ( unsigned int order, unsigned int problem );
    ~FEM();
    double xi_at_node( unsigned int dealNode );
    double basis_function( unsigned int node, double xi );
    double basis_gradient( unsigned int node, double xi );
    void generate_mesh( unsigned int numberOfElements );
    void define_boundary_conds();
    void setup_system();
    void assemble_system();
    void solve();
    void output_results();
    double l2norm_of_error();

    Triangulation<dim>   triangulation;
    FESystem<dim>        fe;
    DoFHandler<dim>      dof_handler;

    unsigned int	      quadRule;
    std::vector<double>	quad_points;
    std::vector<double>	quad_weight;
      
    SparsityPattern       sparsity_pattern;
    SparseMatrix<double>  K;
    Vector<double>        D, F;
    std::vector<double>   nodeLocation;
    std::map<unsigned int,double> boundary_values;
    double                basisFunctionOrder, prob, L, g1, g2, E, h, f;

    std::vector<std::string> nodal_solution_names;
    std::vector< DataComponentInterpretation::DataComponentInterpretation > nodal_data_component_interpretation;
};

template <int dim>
FEM<dim>::FEM( unsigned int order, unsigned int problem ) :
fe( FE_Q<dim>(order), dim ), dof_handler( triangulation )
{
    basisFunctionOrder = order;
    if ( problem == 1 || problem == 2 )
    {
      prob = problem;
    }
    else
    {
      std::cout << "Error: problem number should be 1 or 2.\n";
      exit(0);
    }

    for ( unsigned int i=0; i<dim; ++i )
    {
      nodal_solution_names.push_back("u");
      nodal_data_component_interpretation.push_back( DataComponentInterpretation::component_is_part_of_vector );
    }
}

template <int dim>
FEM<dim>::~FEM()
{
    dof_handler.clear();
}

template <int dim>
double FEM<dim>::xi_at_node( unsigned int dealNode )
{
    double xi;

    if( dealNode == 0 )
    {
      xi = -1.;
    }
    else if( dealNode == 1 )
    {
      xi = 1.;
    }
    else if( dealNode <= basisFunctionOrder )
    {
      xi = -1. + 2.* ( dealNode-1. ) / basisFunctionOrder;
    }
    else
    {
      std::cout << "Error: you input node number "
          << dealNode << " but there are only " 
          << basisFunctionOrder + 1 << " nodes in an element.\n";
      exit(0);
  }

  return xi;
}

template <int dim>
double FEM<dim>::basis_function( unsigned int node, double xi )
{
    double value = 1.;

    for ( unsigned int i = 0; i <= basisFunctionOrder; ++i ) 
    {
      if ( i != node ) 
      {
        value *= ( xi - xi_at_node(i) ) / ( xi_at_node(node) - xi_at_node(i) );
      }
    }
    return value;
}

template <int dim>
double FEM<dim>::basis_gradient( unsigned int node, double xi )
{
    double value = 0.;

    double xi0, xi1, xi2, xi3;
    switch ( (int)basisFunctionOrder )
    {
      case 1:
        switch ( node )
        {
          case 0: 
            value = -1./2.;
            break;
          case 1:
            value = 1./2;
            break;
          default:
            exit(-1);
        }
        break;
      
      case 3:
        switch ( node ) 
        {
          case 3:
            xi1=xi_at_node(0);
            xi2=xi_at_node(1);
            xi3=xi_at_node(2);
            break;
          default:
            xi3=xi_at_node(3);
            break;
        }
      
      case 2:
        xi0 = xi_at_node(node);
        switch ( node )
        {
            case 0:
              xi1=xi_at_node(1);
              xi2=xi_at_node(2);
              break;
          case 1:
              xi1=xi_at_node(0);
              xi2=xi_at_node(2);
              break;
          case 2:
              xi1=xi_at_node(0);
              xi2=xi_at_node(1);
              break;
          default:
            break;
        }
        break;
  
      default:
        break;
    }

    double divider = ( xi0 - xi1 ) * ( xi0 - xi2 );
    if ( basisFunctionOrder == 2 )
    {
        value = ( ( xi - xi2 ) / divider ) + ( ( xi - xi1 ) / divider );
    }

    // Для численого поиска производной используем интерполяционный полином Лагранжа для трех узлов
    if ( basisFunctionOrder == 3 )
    {
        divider *= ( xi0 - xi3 );
        value = ( ( xi - xi2 ) * ( xi - xi3 ) / divider ) + ( ( xi - xi1 ) * ( xi - xi3 ) / divider )+ ( ( xi - xi1 ) * (xi - xi2 ) / divider );
    }
    return value;
}

template <int dim>
void FEM<dim>::generate_mesh( unsigned int numberOfElements )
{
    L = 0.1;
    double x_min = 0.;
    double x_max = L;

    Point< dim, double > min( x_min ), max( x_max );
    std::vector< unsigned int > meshDimensions( dim,numberOfElements );
    GridGenerator::subdivided_hyper_rectangle( triangulation, meshDimensions, min, max );
}

template <int dim>
void FEM<dim>::define_boundary_conds()
{
    const unsigned int totalNodes = dof_handler.n_dofs();

    for( unsigned int globalNode=0; globalNode < totalNodes; ++globalNode )
    {
      if ( nodeLocation[globalNode] == 0 )
      {
        boundary_values[globalNode] = g1;
      }
      
      if ( nodeLocation[globalNode] == L )
      {
        if ( prob == 1 )
        {
          boundary_values[globalNode] = g2;
        }
      }
    }
        
}

template <int dim>
void FEM<dim>::setup_system()
{
    g1 = 0.; g2 = 0.001;
    
    E = 10e+11;
    h = 10e+10;
    f = 10e+11;

    dof_handler.distribute_dofs( fe );

    MappingQ1< dim, dim > mapping;
    std::vector< Point < dim, double > > dof_coords( dof_handler.n_dofs() );
    nodeLocation.resize( dof_handler.n_dofs() );
    DoFTools::map_dofs_to_support_points< dim, dim >( mapping, dof_handler, dof_coords );

    for( unsigned int i=0; i < dof_coords.size(); ++i )
    {
      nodeLocation[i] = dof_coords[i][0];
    }

    define_boundary_conds();

    sparsity_pattern.reinit( dof_handler.n_dofs(), dof_handler.n_dofs(), dof_handler.max_couplings_between_dofs() );
    DoFTools::make_sparsity_pattern( dof_handler, sparsity_pattern );
    sparsity_pattern.compress();
    K.reinit( sparsity_pattern );
    F.reinit( dof_handler.n_dofs() );
    D.reinit( dof_handler.n_dofs() );

    // Выбираем из правила, что квадратура точна для 2n-1 степени многочлена, а мы апроксимируем в случае klocal (N*N) и flocal (f*N) (т.е. максимум 6)
    quadRule = ceil( ( ( basisFunctionOrder * 2 ) + 1 ) / 2 );
    quad_points.resize(quadRule); quad_weight.resize(quadRule);

    switch ((int)quadRule)
    {
        case 1:
            quad_points[0] = 0;
            quad_weight[0] = 2.;
            break;

        case 2:
            quad_points[0] = -sqrt(1. / 3.);
            quad_points[1] = sqrt(1. / 3.);
            //
            quad_weight[0] = 1.;
            quad_weight[1] = 1.;
            break;
        case 3:
            quad_points[0] = 0;
            quad_points[1] = -0.7745966692414834;
            quad_points[2] = 0.7745966692414834;
            //
            quad_weight[0] = 0.88888888;
            quad_weight[1] = 0.555555556;
            quad_weight[2] = 0.5555555556;
            break;
        case 4:
            quad_points[0] = -0.3399810435848563;
            quad_points[1] = 0.3399810435848563;
            quad_points[2] = -0.8611363115940526;
            quad_points[3] = 0.8611363115940526;
            //
            quad_weight[0] = 0.6521451548625461;
            quad_weight[1] = 0.6521451548625461;
            quad_weight[2] = 0.3478548451374538;
            quad_weight[3] = 0.3478548451374538;
            break;
        default:
            quad_points[0] = 0;
            quad_weight[0] = 2.;
            break;
    }
}

template <int dim>
void FEM<dim>::assemble_system()
{
    K=0; F=0;

    const unsigned int          dofs_per_elem = fe.dofs_per_cell;
    FullMatrix<double>          Klocal( dofs_per_elem, dofs_per_elem );
    Vector<double>              Flocal( dofs_per_elem );
    std::vector<unsigned int>   local_dof_indices( dofs_per_elem );
    double                      h_e, x, f;

    typename DoFHandler< dim >::active_cell_iterator elem = dof_handler.begin_active(), endc = dof_handler.end();
    for ( ; elem != endc; ++elem )
    {
      elem->get_dof_indices (local_dof_indices);

      h_e = nodeLocation[local_dof_indices[1]] - nodeLocation[local_dof_indices[0]];

      Flocal = 0.;
      for( unsigned int A = 0; A < dofs_per_elem; ++A ) 
      {
        for( unsigned int q = 0; q < quadRule; ++q )
        {
          x = 0;
          for( unsigned int B = 0; B < dofs_per_elem; ++B )
          {
            x += nodeLocation[local_dof_indices[B]] * basis_function( B, quad_points[q] );
          }

          f *= x;
          Flocal[A] +=  ( h_e / 2. ) * ( basis_function( A, quad_points[q] ) * quad_weight[q] * f );
        }
      }

      if ( prob == 2 )
      { 
        if ( nodeLocation[local_dof_indices[1]] == L )
        {
          Flocal[1] += h;
        }
      }

      Klocal = 0;
      for( unsigned int A = 0; A < dofs_per_elem; ++A ) 
        for( unsigned int B = 0; B < dofs_per_elem; ++B )
          for( unsigned int q = 0; q < quadRule; ++q ) 
            Klocal[A][B] += ( ( 2. * E ) / h_e ) * ( basis_gradient( A, quad_points[q] ) * basis_gradient( B, quad_points[q] ) * quad_weight[q] );
            
      for( unsigned int A = 0; A < dofs_per_elem; ++A ) 
      {
        F[local_dof_indices[A]] += Flocal[A];
        for( unsigned int B = 0; B < dofs_per_elem; ++B ) 
        {
          K.add(local_dof_indices[A], local_dof_indices[B], Klocal[A][B]);
        }
      }
    }
    MatrixTools::apply_boundary_values( boundary_values, K, D, F, false );
}

template <int dim>
void FEM<dim>::solve()
{
    SparseDirectUMFPACK A;
    A.initialize( K );
    A.vmult( D, F );
}

template <int dim>
void FEM< dim >::output_results()
{
    std::ofstream output1( std::string("solution_") + std::to_string((int)prob) + "_" + std::to_string((int)basisFunctionOrder) + std::string(".vtk") );
    DataOut<dim> data_out;
    data_out.attach_dof_handler( dof_handler );

    data_out.add_data_vector( D, nodal_solution_names, DataOut<dim>::type_dof_data, nodal_data_component_interpretation );
    data_out.build_patches();
    data_out.write_vtk( output1 );
    output1.close();
}

template <int dim>
double FEM<dim>::l2norm_of_error()
{
  double l2norm = 0.;

  const unsigned int            dofs_per_elem = fe.dofs_per_cell;
  std::vector<unsigned int>     local_dof_indices( dofs_per_elem );
  double                        u_exact, u_h, x, h_e;

  typename DoFHandler< dim >::active_cell_iterator elem = dof_handler.begin_active(), endc = dof_handler.end();
  for( ; elem != endc; ++ elem )
  {
    elem->get_dof_indices (local_dof_indices);
    h_e = nodeLocation[local_dof_indices[1]] - nodeLocation[local_dof_indices[0]];

    for( unsigned int q = 0; q < quadRule; ++q )
    {
      x = 0.; u_h = 0.;
      for( unsigned int B = 0; B < dofs_per_elem; ++B ) 
      {
        x += nodeLocation[local_dof_indices[B]]*basis_function(B,quad_points[q]);
        u_h += D[local_dof_indices[B]]*basis_function(B,quad_points[q]);
      }
      if ( prob == 1 ) u_exact = ( ( -x * x * x * f ) / (6. * E) ) + ( ( g2 - g1 + ( L * L * L * f ) / ( 6. * E ) ) / L ) * x + g1;
      if ( prob == 2 ) u_exact = -f * ( (x * x * x) / ( 6. * E ) ) + ( ( h + 0.5 * L * L * f ) / E ) * x + g1;

      l2norm += ( u_h - u_exact ) * ( u_h - u_exact ) * ( h_e / 2. ) * quad_weight[q];
    }
  }

  return sqrt(l2norm);
}
