<?php

include("lib/genann.php");

echo("GENANN example 1.".PHP_EOL);
echo("Train a small ANN to the XOR function using backpropagation.".PHP_EOL);

/* This will make the neural network initialize differently each run. */
/* If you don't get a good result, try again for a different result. */
srand( time() );

/* Input and expected out data for the XOR function. */

$input  = FFI::new('double[4][2]');
$output = FFI::new('double[4]');

foreach( [ [ 0 , 0 ] , [ 0 , 1 ] , [ 1 , 0 ] , [ 1 , 1 ] ] as $_y => $_sub ) // TODO FIXME there must be easier way to init FFI arrays
{
	foreach( $_sub as $_x => $_val )
	{
		$input[ $_y ][ $_x ] = $_val ;
	}
}

foreach( [ 0 , 1 , 1 , 0 ] as $_x => $_val )
{
	$output[ $_x ] = $_val ;
}

/* New network with 2 inputs,
 * 1 hidden layer of 2 neurons,
 * and 1 output. */
$ann = Genann::init(2, 1, 2, 1);

/* Train on the four labeled data points many times. */
for( $i = 0 ; $i < 5000 ; $i++ ) 
{
	Genann::train( $ann , $input[0] , $output + 0 , 3 );
	Genann::train( $ann , $input[1] , $output + 1 , 3 );
	Genann::train( $ann , $input[2] , $output + 2 , 3 );
	Genann::train( $ann , $input[3] , $output + 3 , 3 );
}

/* Run the network and see what it predicts. */

for( $i = 0 ; $i < 4 ; $i ++ )
{
	$A = $input[$i][0] ;
	$B = $input[$i][1] ;
	$X = Genann::run( $ann , $input[$i] )[0] ;

	echo( "Output for [ $A , $B ] is $X".PHP_EOL );
}

Genann::free( $ann );

// EOF
