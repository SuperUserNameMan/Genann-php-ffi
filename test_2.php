<?php

include("lib/genann.php");

echo("GENANN example 2.".PHP_EOL);
echo("Train a small ANN to the XOR function using random search.".PHP_EOL);

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

$err = 0.0 ;
$last_err = 1000 ;
$count = 0 ;

do 
{
	$count++;

	if ( $count % 1000 == 0 ) 
	{
		/* We're stuck, start over. */
		Genann::randomize( $ann );
		$last_err = 1000;
	}

	$save = Genann::copy( $ann );

	/* Take a random guess at the ANN weights. */
	for( $i = 0 ; $i < $ann->total_weights ; $i++ ) 
	{
		$ann->weight[ $i ] += rand() / getrandmax() - 0.5;
	}

	/* See how we did. */
	$err = 0;
	$err += pow( Genann::run( $ann , $input[0] )[0] - $output[0] , 2.0 );
	$err += pow( Genann::run( $ann , $input[1] )[0] - $output[1] , 2.0 );
	$err += pow( Genann::run( $ann , $input[2] )[0] - $output[2] , 2.0 );
	$err += pow( Genann::run( $ann , $input[3] )[0] - $output[3] , 2.0 );

	/* Keep these weights if they're an improvement. */
	if ( $err < $last_err )
	{
		Genann::free( $save );
		$last_err = $err;
	} 
	else 
	{
		Genann::free( $ann );
		$ann = $save;
	}

} while ( $err > 0.01 );

echo( "Finished in $count loops.".PHP_EOL );

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
