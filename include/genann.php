<?php 

Genann::Genann(); // autoinit


class Genann 
{
//----------------------------------------------------------------------------------
	// FFI initialisation
	//----------------------------------------------------------------------------------

	public static $ffi;

	public static function Genann()
	{
		if ( static::$ffi ) 
		{ 
			debug_print_backtrace();
			exit("Genann::Genann() already init".PHP_EOL); 
		}
		
		$cdef = __DIR__ . '/genann.ffi.php.h';
		
		$lib_dir = defined('FFI_LIB_DIR') ? FFI_LIB_DIR : 'lib' ;
		
		$slib = "./$lib_dir/libgenann.".PHP_SHLIB_SUFFIX;
		
		static::$ffi = FFI::cdef( file_get_contents( $cdef ) , $slib );
	}


	public static function __callStatic( string $method , array $args ) : mixed
	{
		$callable = [static::$ffi, 'genann_'.$method];
		return $callable(...$args);
	}
	
	//----------------------------------------------------------------------------------
	// Helpers
	//----------------------------------------------------------------------------------

	public static function read( stream $fin ) : object|null
	{
		$header = fgets( $fin );

		list( $inputs , $hidden_layers , $hidden , $outputs ) = explode( ' ' , $header );

		$ann = static::init( intval( $inputs ) , intval( $hidden_layers ) , intval( $hidden ) , intval( $outputs ) );

		for( $i = 0 ; $i < $ann->total_weights ; $i++ )
		{
			$ann->weight[ $i ] = floatval( fgets( $fin ) );
		}
	}

	public static function write( object $ann , stream $fout )
	{
		fwrite( $fout , $ann->inputs.' '.$ann->hidden_layers.' '.$ann->hidden.' '.$ann->outputs.PHP_EOL );

		for( $i = 0 ; $i < $ann->total_weights ; $i++ )
		{
			fwrite( $fout , $ann->weight[ $i ].PHP_EOL );
		}
	}
};

// EOF
